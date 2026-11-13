// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <EGL.h>
#include <PTexLib.h>
#include <pangolin/image/image_convert.h>

#include <fstream>
#include <sstream>

#include "GLCheck.h"
#include "MirrorRenderer.h"

// Read camera poses in the format of NICE-SLAM's traj.txt: one camera-to-world
// 4x4 matrix per line, as 16 numbers in row-major order, with the camera axes
// of this renderer and OpenCV (x right, y down, z forward).
static std::vector<Eigen::Matrix4d> LoadPoses(const std::string& poseFile) {
  std::vector<Eigen::Matrix4d> poses;
  std::ifstream file(poseFile);
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream values(line);
    Eigen::Matrix4d T_world_camera;
    bool ok = true;
    for (int r = 0; r < 4 && ok; r++) {
      for (int c = 0; c < 4 && ok; c++) {
        ok = static_cast<bool>(values >> T_world_camera(r, c));
      }
    }
    if (ok) {
      poses.push_back(T_world_camera);
    } else {
      // blank lines are fine; anything else is a malformed pose
      ASSERT(line.find_first_not_of(" \t\r") == std::string::npos,
             "Expected 16 numbers per line in " + poseFile);
    }
  }
  return poses;
}


int main(int argc, char* argv[]) {
  ASSERT(argc >= 3 && argc <= 5,
         "Usage: ./ReplicaRenderer mesh.ply /path/to/atlases [mirrorFile [poseFile]]\n"
         "  mirrorFile  glass.sur; pass \"\" for none when giving a poseFile\n"
         "  poseFile    camera-to-world poses, 16 numbers per line (NICE-SLAM traj.txt);\n"
         "              one frame is rendered per pose instead of the built-in trajectory");

  const std::string meshFile(argv[1]);
  const std::string atlasFolder(argv[2]);
  ASSERT(pangolin::FileExists(meshFile));
  ASSERT(pangolin::FileExists(atlasFolder));

  std::string surfaceFile;
  if (argc >= 4 && argv[3][0] != '\0') {
    surfaceFile = std::string(argv[3]);
    ASSERT(pangolin::FileExists(surfaceFile));
  }

  std::vector<Eigen::Matrix4d> poses;
  if (argc == 5) {
    const std::string poseFile(argv[4]);
    ASSERT(pangolin::FileExists(poseFile));
    poses = LoadPoses(poseFile);
    ASSERT(!poses.empty(), "No poses in " + poseFile);
    std::cout << "Loaded " << poses.size() << " poses" << std::endl;
  }

  const int width = 1280;
  const int height = 960;
  bool renderDepth = true;
  float depthScale = 65535.0f * 0.1f;

  // Setup EGL
  EGLCtx egl;

  egl.PrintInformation();
  
  if(!checkGLVersion()) {
    return 1;
  }

  //Don't draw backfaces
  const GLenum frontFace = GL_CCW;
  glFrontFace(frontFace);

  // Setup a framebuffer
  pangolin::GlTexture render(width, height);
  pangolin::GlRenderBuffer renderBuffer(width, height);
  pangolin::GlFramebuffer frameBuffer(render, renderBuffer);

  pangolin::GlTexture depthTexture(width, height, GL_R32F, false, 0, GL_RED, GL_FLOAT, 0);
  pangolin::GlFramebuffer depthFrameBuffer(depthTexture, renderBuffer);

  // Setup a camera
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixRDF_BottomLeft(
          width,
          height,
          width / 2.0f,
          width / 2.0f,
          (width - 1.0f) / 2.0f,
          (height - 1.0f) / 2.0f,
          0.1f,
          100.0f),
      pangolin::ModelViewLookAtRDF(0, 0, 4, 0, 0, 0, 0, 1, 0));

  // Start at some origin
  Eigen::Matrix4d T_camera_world = s_cam.GetModelViewMatrix();

  // And move to the left
  Eigen::Matrix4d T_new_old = Eigen::Matrix4d::Identity();

  T_new_old.topRightCorner(3, 1) = Eigen::Vector3d(0.025, 0, 0);

  // load mirrors
  std::vector<MirrorSurface> mirrors;
  if (surfaceFile.length()) {
    std::ifstream file(surfaceFile);
    picojson::value json;
    picojson::parse(json, file);

    for (size_t i = 0; i < json.size(); i++) {
      mirrors.emplace_back(json[i]);
    }
    std::cout << "Loaded " << mirrors.size() << " mirrors" << std::endl;
  }

  const std::string shadir = STR(SHADER_DIR);
  MirrorRenderer mirrorRenderer(mirrors, width, height, shadir);

  // load mesh and textures
  PTexMesh ptexMesh(meshFile, atlasFolder);

  pangolin::ManagedImage<Eigen::Matrix<uint8_t, 3, 1>> image(width, height);
  pangolin::ManagedImage<float> depthImage(width, height);
  pangolin::ManagedImage<uint16_t> depthImageInt(width, height);

  // Render some frames
  const size_t numFrames = poses.empty() ? 100 : poses.size();
  for (size_t i = 0; i < numFrames; i++) {
    if (!poses.empty()) {
      T_camera_world = poses[i].inverse();
      s_cam.GetModelViewMatrix() = T_camera_world;
    }

    std::cout << "\rRendering frame " << i + 1 << "/" << numFrames << "... ";
    std::cout.flush();

    // Render
    frameBuffer.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, width, height);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_CULL_FACE);

    ptexMesh.Render(s_cam);

    glDisable(GL_CULL_FACE);

    glPopAttrib(); //GL_VIEWPORT_BIT
    frameBuffer.Unbind();

    for (size_t i = 0; i < mirrors.size(); i++) {
      MirrorSurface& mirror = mirrors[i];
      // capture reflections
      mirrorRenderer.CaptureReflection(mirror, ptexMesh, s_cam, frontFace);

      frameBuffer.Bind();
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);

      // render mirror
      mirrorRenderer.Render(mirror, mirrorRenderer.GetMaskTexture(i), s_cam);

      glPopAttrib(); //GL_VIEWPORT_BIT
      frameBuffer.Unbind();
    }

    // Download and save
    render.Download(image.ptr, GL_RGB, GL_UNSIGNED_BYTE);

    char filename[1000];
    snprintf(filename, 1000, "frame%06zu.jpg", i);

    pangolin::SaveImage(
        image.UnsafeReinterpret<uint8_t>(),
        pangolin::PixelFormatFromString("RGB24"),
        std::string(filename));

    if (renderDepth) {
      // render depth
      depthFrameBuffer.Bind();
      glPushAttrib(GL_VIEWPORT_BIT);
      glViewport(0, 0, width, height);
      glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

      glEnable(GL_CULL_FACE);

      ptexMesh.RenderDepth(s_cam, depthScale);

      glDisable(GL_CULL_FACE);

      glPopAttrib(); //GL_VIEWPORT_BIT
      depthFrameBuffer.Unbind();

      depthTexture.Download(depthImage.ptr, GL_RED, GL_FLOAT);

      // convert to 16-bit int
      for(size_t i = 0; i < depthImage.Area(); i++)
          depthImageInt[i] = static_cast<uint16_t>(depthImage[i] + 0.5f);

      snprintf(filename, 1000, "depth%06zu.png", i);
      pangolin::SaveImage(
          depthImageInt.UnsafeReinterpret<uint8_t>(),
          pangolin::PixelFormatFromString("GRAY16LE"),
          std::string(filename), true, 34.0f);
    }

    if (poses.empty()) {
      // Move the camera
      T_camera_world = T_camera_world * T_new_old.inverse();

      s_cam.GetModelViewMatrix() = T_camera_world;
    }
  }
  std::cout << "\rRendering frame " << numFrames << "/" << numFrames << "... done" << std::endl;

  return 0;
}


// ch05 使用freeglut 来操作opengl -- 动态三角形
#include <iostream>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "ShaderReader.h"
#include <Eigen\Core>
#include <vector>

#include "Camera.h"
#include "MeshManager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

unsigned int VBO, VAO, EBO;
ShaderReader lightShader, sceneShader, blurShader, finalShader;
ShaderReader deferredLightShader, deferredShadingShader, fboDebugShader, gBufferShader;
Camera  sceneCamera, lightCamera;
MeshMananger meshManager;
unsigned int gScaleLocation;

float scale = 0.1f;
float near_plane = 1, far_plane = 100;
unsigned int texture_wall, texture_disp, texture_normal;
Matrix4f modelMatrix, translateMatrix, rotateMatrix, scaleMatrix, lightSpaceMatrix;
std::string strTextureLocation;

unsigned int depthMapFBO,depthMap;
const int SCR_WIDTH = 600, SCR_HEIGHT = 600;
const int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;

bool renderDepth = true;

std::vector<Vector3f> lightPositions, lightColors;
unsigned int hdrFBO, rboDepth, colorBuffers[2],  pingpongFBO[2], pingpongColorBuffers[2];

int bloom = 1;
float exposure = 2.0f;

unsigned int gBuffer, gPosition, gNormal, gAlbedoSpec;

void RenderScene(ShaderReader whichShader)
{
    translateMatrix = sceneCamera.GetTranslateMatirx(0, -1, 0);
    scaleMatrix = sceneCamera.GetScaleMatrix(3, 0.1, 3);
    rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
    modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
    whichShader.setMatrix4("model", modelMatrix);
    meshManager.RenderCube();

    translateMatrix = sceneCamera.GetTranslateMatirx(1, 0, 2);
    scaleMatrix = sceneCamera.GetScaleMatrix(0.5, 1, 0.5);
    rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
    modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
    whichShader.setMatrix4("model", modelMatrix);
    meshManager.RenderCube();

    translateMatrix = sceneCamera.GetTranslateMatirx(1, 0, -2);
    scaleMatrix = sceneCamera.GetScaleMatrix(0.5, 1, 0.5);
    rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
    modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
    whichShader.setMatrix4("model", modelMatrix);
    meshManager.RenderCube();
}

void RenderMainProgram() {


    //========= 第一步：渲染几何场景==================
    glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
    gBufferShader.use();
    gBufferShader.setMatrix4("projection", sceneCamera.PerspectiveMatrix);
    gBufferShader.setMatrix4("view", sceneCamera.lookAt());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_wall);
    strTextureLocation = "texture_diffuse1";
    glUniform1i(glGetUniformLocation(gBufferShader.ID, strTextureLocation.c_str()), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture_disp);
    strTextureLocation = "texture_specular1";
    glUniform1i(glGetUniformLocation(gBufferShader.ID, strTextureLocation.c_str()), 1);
    RenderScene(gBufferShader);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //========= 第二步：处理光照==================
    deferredShadingShader.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gPosition);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, gNormal);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
    // send light relevant uniforms
    for (unsigned int i = 0; i < lightPositions.size(); i++)
    {
        deferredShadingShader.setVec3("lights[" + std::to_string(i) + "].Position", lightPositions[i](0), lightPositions[i](1), lightPositions[i](2));
        deferredShadingShader.setVec3("lights[" + std::to_string(i) + "].Color", lightColors[i](0), lightColors[i](1), lightColors[i](2));
        // update attenuation parameters and calculate radius
        const float linear = 0.7;
        const float quadratic = 1.8;
        deferredShadingShader.setFloat("lights[" + std::to_string(i) + "].Linear", linear);
        deferredShadingShader.setFloat("lights[" + std::to_string(i) + "].Quadratic", quadratic);
    }
    deferredShadingShader.setVec3("viewPos", sceneCamera.Position(0), sceneCamera.Position(1), sceneCamera.Position(2));
    meshManager.RenderQuad();
    //========= 第三步：重新渲染几何场景==================
    glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, SCR_WIDTH, SCR_HEIGHT, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    //=========第四步：将灯光渲染出来==================
    deferredLightShader.use();
    deferredLightShader.setMatrix4("projection", sceneCamera.PerspectiveMatrix);
    deferredLightShader.setMatrix4("view", sceneCamera.lookAt());
    for (unsigned int i = 0; i < lightPositions.size(); i++)
    {
        translateMatrix = sceneCamera.GetTranslateMatirx(lightPositions[i](0), lightPositions[i](1), lightPositions[i](2));
        scaleMatrix = sceneCamera.GetScaleMatrix(0.125, 0.125, 0.125);
        rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
        modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
        deferredLightShader.setMatrix4("model", modelMatrix);
        deferredLightShader.setVec3("lightColor", lightColors[i](0), lightColors[i](1), lightColors[i](2));
        meshManager.RenderCube();
    }
    glutSwapBuffers();
}

int loadTexture(char* ImagePath)
{
    unsigned int textureId;
    glGenTextures(1, &textureId);
    glBindTexture(GL_TEXTURE_2D, textureId); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
    // set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load image, create texture and generate mipmaps
    int width, height, nrChannels;
    // The FileSystem::getPath(...) is part of the GitHub repository so we can find files on any IDE/platform; replace it with your own image path.
    unsigned char* data = stbi_load(ImagePath, &width, &height, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
    return textureId;
}

void process_Normal_Keys(int key, int x, int y)
{
    switch (key)
    {
    case 27:      break;
    case 100: printf("lightdirx %f\n", sceneCamera.Position(0)); sceneCamera.Position(0) += 0.1;   break;
    case 102: printf("lightdirx %f\n", sceneCamera.Position(0)); sceneCamera.Position(0) -= 0.1;  break;
    case 101: printf("lightdiry %f\n", sceneCamera.Position(2)); sceneCamera.Position(2) += 0.1;  break;
    case 103: printf("lightdiry %f\n", sceneCamera.Position(2)); sceneCamera.Position(2) -= 0.1;  break;
    }

}

int main(int argc, char** argv) {

    // ================== 初始化glut  ==========================
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(SCR_HEIGHT, SCR_WIDTH); 
    glutInitWindowPosition(100, 100); 
    glutCreateWindow("Tutorial 01");  
    glutDisplayFunc(RenderMainProgram);
    glutIdleFunc(RenderMainProgram);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glewInit();

    // ================== 着色器相关 ==========================
    /*
    lightShader.ReadShaderFile("simple.vert", "simple.frag");
    sceneShader.ReadShaderFile("screen.vert", "screen.frag");
    blurShader.ReadShaderFile("blur.vert", "blur.frag");
    finalShader.ReadShaderFile("blur.vert", "bloom_final.frag");
    */

    deferredLightShader.ReadShaderFile("deferred_light_box.vert", "deferred_light_box.frag");
    deferredShadingShader.ReadShaderFile("deferred_shading.vert", "deferred_shading.frag");
    gBufferShader.ReadShaderFile("g_buffer.vert", "g_buffer.frag");

    gBufferShader.use();
    gBufferShader.setInt("texture_diffuse1", 0);
    gBufferShader.setInt("texture_specular1", 1);

    deferredShadingShader.use();
    deferredShadingShader.setInt("gPosition", 0);
    deferredShadingShader.setInt("gNormal", 1);
    deferredShadingShader.setInt("gAlbedoSpec", 2);

    gScaleLocation = glGetUniformLocation(lightShader.ID, "gScale");

    // ================== 设置摄像机 ==========================
    sceneCamera = Camera(4.0f, 4.0f, 6.0f, 0.0, 0.0, 0.0);
    sceneCamera.SetPerspectiveMatrix2(-0.5, 0.5, 0.5, -0.5, 1, 100);
    lightCamera = Camera(6.0f, 4.0f, 0.0f, 0.0, 0.0, 0.0);
    lightCamera.SetPerspectiveMatrix2(-0.5, 0.5, 0.5, -0.5, 1, 100);

    // ================== 加载图片 ==========================
    std::string strImage = "bricks2.jpg";
    char* imagePath = const_cast<char*>(strImage.c_str());
    texture_wall = loadTexture(imagePath);
    strImage = "bricks2_disp.jpg";
    texture_disp = loadTexture(const_cast<char*>(strImage.c_str()));
    strImage = "bricks2_normal.jpg";
    texture_normal = loadTexture(const_cast<char*>(strImage.c_str()));

    // ================== 阴影的FrameBuffer ==========================
    glGenFramebuffers(1, &depthMapFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    //glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, depthMap, 0);
    //glDrawBuffer(GL_NONE);
    //glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // ================== gBuffer ==========================
    glGenFramebuffers(1, &gBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);
    // position color buffer
    glGenTextures(1, &gPosition);
    glBindTexture(GL_TEXTURE_2D, gPosition);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);
    // normal color buffer
    glGenTextures(1, &gNormal);
    glBindTexture(GL_TEXTURE_2D, gNormal);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);
    // color + specular color buffer
    glGenTextures(1, &gAlbedoSpec);
    glBindTexture(GL_TEXTURE_2D, gAlbedoSpec);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, gAlbedoSpec, 0);
    // tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
    unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers(3, attachments);
    //==================深度RenderBuffer==============
    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //================== 灯光 =====================
    for (unsigned int i = 0; i < 16; i++)
    {
        // calculate slightly random offsets
        float xPos = ((rand() % 100) / 100.0) * 6.0 - 3.0;
        float yPos = ((rand() % 100) / 100.0) * 6.0 - 4.0;
        float zPos = ((rand() % 100) / 100.0) * 6.0 - 3.0;
        lightPositions.push_back(Eigen::Vector3f(xPos, yPos, zPos));
        // also calculate random color
        float rColor = ((rand() % 100) / 200.0f) + 0.5; // between 0.5 and 1.0
        float gColor = ((rand() % 100) / 200.0f) + 0.5; // between 0.5 and 1.0
        float bColor = ((rand() % 100) / 200.0f) + 0.5; // between 0.5 and 1.0
        lightColors.push_back(Eigen::Vector3f(rColor, gColor, bColor));
    }
    // ================== glut开始执行  ==========================
    glutSpecialFunc(process_Normal_Keys);
    glutMainLoop();

    return 0;
}
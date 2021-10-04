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
unsigned int hdrFBO, rboDepth, colorBuffers[2], attachments[2], pingpongFBO[2], pingpongColorBuffers[2];

int bloom = 1;
float exposure = 2.0f;

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

void RenderShadow()
{
    // 清空颜色缓存





}

void RenderMainProgram() {

    //========= 第一步：深度贴图==================
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    lightShader.use();
    scale -= 0.0001f;
    glUniform1f(gScaleLocation, scale);
    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    lightShader.setFloat("near_plane", near_plane);
    lightShader.setFloat("far_plane", far_plane);
    lightShader.setMatrix4("projection", lightCamera.PerspectiveMatrix);
    lightShader.setMatrix4("view", lightCamera.lookAt());
    lightSpaceMatrix = lightCamera.PerspectiveMatrix * lightCamera.lookAt();
    RenderScene(lightShader);
    //========= 第二步：渲染正常场景==================
    glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
    sceneShader.use();
    sceneShader.setMatrix4("lightSpaceMatrix", lightSpaceMatrix);
    sceneShader.setVec3("lightPos", lightCamera.Position(0), lightCamera.Position(1), lightCamera.Position(2));
    sceneShader.setVec3("cameraPos", sceneCamera.Position(0), sceneCamera.Position(1), sceneCamera.Position(2));
    sceneShader.setMatrix4("projection", sceneCamera.PerspectiveMatrix);
    sceneShader.setMatrix4("view", sceneCamera.lookAt());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    strTextureLocation = "depthMap";
    glUniform1i(glGetUniformLocation(sceneShader.ID, strTextureLocation.c_str()), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture_wall);
    strTextureLocation = "diffuseMap";
    glUniform1i(glGetUniformLocation(sceneShader.ID, strTextureLocation.c_str()), 1);
    RenderScene(sceneShader);
    //========= 第三步：模糊BlurColor==================
    bool horizontal = true, first_iteration = true;
    unsigned int amount = 10;
    blurShader.use();
    for (unsigned int i = 0; i < amount; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[horizontal]);
        blurShader.setInt("horizontal", horizontal);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, first_iteration ? colorBuffers[1] : pingpongColorBuffers[!horizontal]);  // bind texture of other framebuffer (or scene if first iteration)
        strTextureLocation = "image";
        glUniform1i(glGetUniformLocation(blurShader.ID, strTextureLocation.c_str()), 0);
        meshManager.RenderQuad();
        horizontal = !horizontal;
        if (first_iteration)
            first_iteration = false;
    }
    // ==================== 第四步：渲染最后的图片 =====================
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    finalShader.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, colorBuffers[0]);
    strTextureLocation = "scene";
    glUniform1i(glGetUniformLocation(finalShader.ID, strTextureLocation.c_str()), 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, pingpongColorBuffers[horizontal]);
    strTextureLocation = "bloomBlur";
    glUniform1i(glGetUniformLocation(finalShader.ID, strTextureLocation.c_str()), 1);
    finalShader.setInt("bloom", bloom);
    finalShader.setFloat("exposure", exposure);
    meshManager.RenderQuad();
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
    lightShader.ReadShaderFile("simple.vert", "simple.frag");
    sceneShader.ReadShaderFile("screen.vert", "screen.frag");
    blurShader.ReadShaderFile("blur.vert", "blur.frag");
    finalShader.ReadShaderFile("blur.vert", "bloom_final.frag");
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
    // ================== 高光hdrFBO ==========================
    glGenFramebuffers(1, &hdrFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, hdrFBO);
    glGenTextures(2, colorBuffers);
    for (unsigned int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, colorBuffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  // we clamp to the edge as the blur filter would otherwise sample repeated texture values!
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorBuffers[i], 0);
    }

    // ================== render buffer ==========================
    glGenRenderbuffers(1, &rboDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);
    // tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
    unsigned int attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers(2, attachments);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // ================== 交替Buffer ==========================
    glGenFramebuffers(2, pingpongFBO);
    glGenTextures(2, pingpongColorBuffers);
    for (unsigned int i = 0; i < 2; i++)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, pingpongFBO[i]);
        glBindTexture(GL_TEXTURE_2D, pingpongColorBuffers[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // we clamp to the edge as the blur filter would otherwise sample repeated texture values!
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pingpongColorBuffers[i], 0);
        // also check if framebuffers are complete (no need for depth buffer)
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "Framebuffer not complete!" << std::endl;
    }

    // ================== glut开始执行  ==========================
    glutSpecialFunc(process_Normal_Keys);
    glutMainLoop();

    return 0;
}
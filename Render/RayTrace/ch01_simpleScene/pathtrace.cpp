// ch05 使用freeglut 来操作opengl -- 动态三角形
#include <iostream>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "ShaderReader.h"
#include <Eigen\Core>

#include "Camera.h"
#include "MeshManager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

unsigned int VBO, VAO, EBO;
ShaderReader lightShader, sceneShader;
Camera  sceneCamera, lightCamera;
MeshMananger meshManager;
unsigned int gScaleLocation;

float scale = 0.1f;
float near_plane = 1, far_plane = 100;
unsigned int texture_wall, texture_disp, texture_normal;
Matrix4f modelMatrix, translateMatrix, rotateMatrix, scaleMatrix, lightSpaceMatrix;
std::string strTextureLocation;

unsigned int depthMapFBO, depthMap;
const int SCR_WIDTH = 600, SCR_HEIGHT = 600;
const int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;

bool renderDepth = true;

void RenderScene(ShaderReader whichShader)
{
    /*
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
    */

    modelMatrix = Matrix4f::Identity();
    whichShader.setMatrix4("model", modelMatrix);
    meshManager.RenderQuad();
}

void RenderMainProgram() {
    // 清空颜色缓存
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    //glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
    glViewport(0, 0, SHADOW_WIDTH, SHADOW_HEIGHT);
    lightShader.use();
    scale -= 0.0001f;
    glUniform1f(gScaleLocation, scale);
    if (renderDepth) glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    lightShader.setFloat("near_plane", near_plane);
    lightShader.setFloat("far_plane", far_plane);
    lightShader.setMatrix4("projection", lightCamera.PerspectiveMatrix);
    lightShader.setMatrix4("view", lightCamera.lookAt());
    lightSpaceMatrix = lightCamera.PerspectiveMatrix * lightCamera.lookAt();

    RenderScene(lightShader);



    if (renderDepth)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
        sceneShader.use();
        sceneShader.setMatrix4("lightSpaceMatrix", lightSpaceMatrix);
        sceneShader.setVec3("eyePos", sceneCamera.Position(0), sceneCamera.Position(1), sceneCamera.Position(2));
        sceneCamera.UpdateUpAndRightVector();
        sceneShader.setVec3("eyeDir", sceneCamera.Front(0), sceneCamera.Front(1), sceneCamera.Front(2));
        sceneShader.setVec3("eyeUp", sceneCamera.UpVector(0), sceneCamera.UpVector(1), sceneCamera.UpVector(2));
        sceneShader.setVec3("eyeRight", sceneCamera.RightVector(0), sceneCamera.RightVector(1), sceneCamera.RightVector(2));

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

    }



    // 交换前后缓存
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
    default:printf("FrontVector:%f,%f,%f\nUpVector:%f,%f,%f\nRightVector:%f,%f,%f\n",
        sceneCamera.Front(0), sceneCamera.Front(1), sceneCamera.Front(2),
        sceneCamera.UpVector(0), sceneCamera.UpVector(1), sceneCamera.UpVector(2),
        sceneCamera.RightVector(0), sceneCamera.RightVector(1), sceneCamera.RightVector(2)); break;
    }

}

int main(int argc, char** argv) {

    // 初始化GLUT
    glutInit(&argc, argv);
    // 显示模式：双缓冲、RGBA
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    // 窗口设置
    glutInitWindowSize(SCR_HEIGHT, SCR_WIDTH);      // 窗口尺寸
    glutInitWindowPosition(100, 100);  // 窗口位置
    glutCreateWindow("RayTracing");   // 窗口标题

    glutDisplayFunc(RenderMainProgram);
    glutIdleFunc(RenderMainProgram);

    // 缓存清空后的颜色值
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glewInit();
    lightShader.ReadShaderFile("simple.vert", "simple.frag");
    sceneShader.ReadShaderFile("screen.vert", "screen.frag");

    gScaleLocation = glGetUniformLocation(lightShader.ID, "gScale");


    sceneCamera = Camera(0.0f, 4.0f, -4.0f, 0.0, 0.0, 0.0);
    sceneCamera.SetPerspectiveMatrix2(-0.5, 0.5, 0.5, -0.5, 1, 100);
    lightCamera = Camera(6.0f, 4.0f, 0.0f, 0.0, 0.0, 0.0);
    lightCamera.SetPerspectiveMatrix2(-0.5, 0.5, 0.5, -0.5, 1, 100);


    std::string strImage = "bricks2.jpg";
    char* imagePath = const_cast<char*>(strImage.c_str());
    texture_wall = loadTexture(imagePath);
    strImage = "bricks2_disp.jpg";
    texture_disp = loadTexture(const_cast<char*>(strImage.c_str()));
    strImage = "bricks2_normal.jpg";
    texture_normal = loadTexture(const_cast<char*>(strImage.c_str()));

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

    glutSpecialFunc(process_Normal_Keys);
    glutMainLoop();

    return 0;
}
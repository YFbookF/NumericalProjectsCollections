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
ShaderReader lightShader,sceneShader;
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
    translateMatrix = lightCamera.GetTranslateMatirx(0, -1, 0);
    scaleMatrix = lightCamera.GetScaleMatrix(3, 0.1, 3);
    rotateMatrix = lightCamera.GetRotateMatrix(0, 0, 1, 0);
    modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
    lightShader.setMatrix4("model", modelMatrix);
    meshManager.RenderCube();
    translateMatrix = lightCamera.GetTranslateMatirx(1,0, 0);
    scaleMatrix = lightCamera.GetScaleMatrix(1, 1, 1);
    rotateMatrix = lightCamera.GetRotateMatrix(0, 0, 1, 0);
    modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
    lightShader.setMatrix4("model", modelMatrix);
    meshManager.RenderCube();

    if (renderDepth)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
        sceneShader.use();
        translateMatrix = sceneCamera.GetTranslateMatirx(1, 0, 0);
        scaleMatrix = sceneCamera.GetScaleMatrix(1, 1, 1);
        rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
        modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
        sceneShader.setMatrix4("model", modelMatrix);
        sceneShader.setMatrix4("projection", sceneCamera.PerspectiveMatrix);
        sceneShader.setMatrix4("view", sceneCamera.lookAt());
        sceneShader.setMatrix4("lightSpaceMatrix", lightSpaceMatrix);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        strTextureLocation = "depthMap";
        glUniform1i(glGetUniformLocation(sceneShader.ID, strTextureLocation.c_str()), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        strTextureLocation = "diffuseMap";
        glUniform1i(glGetUniformLocation(sceneShader.ID, strTextureLocation.c_str()), 1);

        meshManager.RenderCube();
        translateMatrix = sceneCamera.GetTranslateMatirx(0, -1, 0);
        scaleMatrix = sceneCamera.GetScaleMatrix(3, 0.1, 3);
        rotateMatrix = sceneCamera.GetRotateMatrix(0, 0, 1, 0);
        modelMatrix = translateMatrix * rotateMatrix * scaleMatrix;
        sceneShader.setMatrix4("model", modelMatrix);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthMap);
        strTextureLocation = "diffuseMap";
        glUniform1i(glGetUniformLocation(sceneShader.ID, strTextureLocation.c_str()), 1);
        meshManager.RenderCube();
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


int main(int argc, char** argv) {

    // 初始化GLUT
    glutInit(&argc, argv);
    // 显示模式：双缓冲、RGBA
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    // 窗口设置
    glutInitWindowSize(SCR_HEIGHT, SCR_WIDTH);      // 窗口尺寸
    glutInitWindowPosition(100, 100);  // 窗口位置
    glutCreateWindow("Tutorial 01");   // 窗口标题

    glutDisplayFunc(RenderMainProgram);
    glutIdleFunc(RenderMainProgram);

    // 缓存清空后的颜色值
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glewInit();
    lightShader.ReadShaderFile("simple.vert", "simple.frag");
    sceneShader.ReadShaderFile("screen.vert", "screen.frag");
    
    gScaleLocation = glGetUniformLocation(lightShader.ID, "gScale");

    float vertices[] = {
        // positions          // colors       
         0.5f,  0.5f, -2.0f,   1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -2.0f,   0.0f, 1.0f, 0.0f,
        -0.5f, -0.5f, -2.0f,   0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f, -2.0f,   1.0f, 1.0f, 0.0f,
    };
    unsigned int indices[] = {  // note that we start from 0!
        0, 1, 3,  // first Triangle
        1, 2, 3   // second Triangle
    };
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    sceneCamera = Camera(0.0f, 4.0f, 6.0f, 0.0, 0.0, 0.0);
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

    // 通知开始GLUT的内部循环
    glutMainLoop();

    return 0;
}
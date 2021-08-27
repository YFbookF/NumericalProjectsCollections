// ch05 使用freeglut 来操作opengl -- 动态三角形
#include <iostream>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include "Camera.h"
#include "ShaderReader.h"

unsigned int VBO, VAO, EBO;
ShaderReader ourShader;
unsigned int gScaleLocation, gWorldToCameraMatrix;
Camera camera;
float scale = 1.0f;
float nearPlane = 100.0f;

void RenderMainProgram() {
    // 清空颜色缓存
    glClear(GL_COLOR_BUFFER_BIT);

    glUniform1f(gScaleLocation, scale);

    
GLfloat world2camera[] = { camera.PerspectiveMatrix(0,0),camera.PerspectiveMatrix(0,1) ,camera.PerspectiveMatrix(0,2),camera.PerspectiveMatrix(0,3) ,
    camera.PerspectiveMatrix(1,0),camera.PerspectiveMatrix(1,1) ,camera.PerspectiveMatrix(1,2),camera.PerspectiveMatrix(1,3) ,
    camera.PerspectiveMatrix(2,0),camera.PerspectiveMatrix(2,1) ,camera.PerspectiveMatrix(2,2),camera.PerspectiveMatrix(2,3) ,
    camera.PerspectiveMatrix(3,0),camera.PerspectiveMatrix(3,1) ,camera.PerspectiveMatrix(3,2),camera.PerspectiveMatrix(3,3) ,
};
/*
GLfloat world2camera2[] = { camera.PerspectiveMatrix(0,0),camera.PerspectiveMatrix(1,0) ,camera.PerspectiveMatrix(2,0),camera.PerspectiveMatrix(3,0) ,
    camera.PerspectiveMatrix(0,1),camera.PerspectiveMatrix(1,1) ,camera.PerspectiveMatrix(2,1),camera.PerspectiveMatrix(3,1) ,
    camera.PerspectiveMatrix(0,2),camera.PerspectiveMatrix(1,2) ,camera.PerspectiveMatrix(2,2),camera.PerspectiveMatrix(3,2) ,
    camera.PerspectiveMatrix(0,3),camera.PerspectiveMatrix(1,3) ,camera.PerspectiveMatrix(2,3),camera.PerspectiveMatrix(3,3) ,
};*/

glUniformMatrix4fv(gWorldToCameraMatrix, 1, GL_TRUE, (const GLfloat*)world2camera);

    ourShader.use();
    glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
           //glDrawArrays(GL_TRIANGLES, 0, 6);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    // 交换前后缓存
    glutSwapBuffers();
}

/**
 * 主函数
 */
int main(int argc, char** argv) {

    // 初始化GLUT
    glutInit(&argc, argv);
    // 显示模式：双缓冲、RGBA
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    // 窗口设置
    glutInitWindowSize(480, 320);      // 窗口尺寸
    glutInitWindowPosition(100, 100);  // 窗口位置
    glutCreateWindow("Tutorial 01");   // 窗口标题

    glutDisplayFunc(RenderMainProgram);
    glutIdleFunc(RenderMainProgram);

    // 缓存清空后的颜色值
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glewInit();
    ourShader.ReadShaderFile("simple.vert", "simple.frag");
    gScaleLocation = glGetUniformLocation(ourShader.ID, "gScale");
    gWorldToCameraMatrix = glGetUniformLocation(ourShader.ID, "worldToCameraMatrix");

    float rect = 1.0f;

    float vertices[] = {
        // positions          // colors       
         rect,  rect, nearPlane,   1.0f, 0.0f, 0.0f,
         rect, -rect, nearPlane,   0.0f, 1.0f, 0.0f,
        -rect, -rect, nearPlane,   0.0f, 0.0f, 1.0f,
        -rect,  rect, nearPlane,   1.0f, 1.0f, 0.0f,
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

    // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    glBindVertexArray(0);
    camera.SetPerspectiveMatrix(400.0f, 400.0f, 0.0f, 1000.0f, 60.0f);

    // 通知开始GLUT的内部循环
    glutMainLoop();

    return 0;
}
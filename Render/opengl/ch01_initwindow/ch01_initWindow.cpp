// ch01 使用freeglut 来操作opengl -- 初始化窗口
// 如出现无法解析的符号glviewport，请添加opengl32.lib
// 此系列代码仅需要glew库和freeglut库
#include <iostream>
#include <GL\freeglut.h>

/**
 * 渲染回调函数
 */
void RenderScenceCB() {
    // 清空颜色缓存
    glClear(GL_COLOR_BUFFER_BIT);
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

    // 开始渲染
    glutDisplayFunc(RenderScenceCB);

    // 缓存清空后的颜色值
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    // 通知开始GLUT的内部循环
    glutMainLoop();

    return 0;
}
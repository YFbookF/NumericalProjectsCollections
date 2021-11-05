#ifndef  __CUDA_RENDER_H_
#define  __CUDA_RENDER_H_


class Vertex;
class Triangle;

#define M_PI 3.14159265359f
#define width 512	// screenwidth
#define height 512	// screenheight
#define samps  1	// samples per pixel per pass



extern unsigned int vertices_num;
extern Vertex* scene_vertices_pos;
extern unsigned int triangle_num;
extern Triangle* scene_triangles;
/*
int total_number_of_scene_triangles = 0;
float* dev_triangle_p; // the cuda device pointer that points to the uploaded scene_triangles
std::vector<float4> cuda_triangles;

bool firstTime = true;

// scene bounding box
float3 scene_aabbox_min;
float3 scene_aabbox_max;

void pre_render_kernel(float3* output, float3* accumbuffer, const int numscene_triangles, int framenumber, int hashedframenumber, float3 scene_bbmin, float3 scene_bbmax);
*/
#endif // ! __CUDA_RENDER_H_

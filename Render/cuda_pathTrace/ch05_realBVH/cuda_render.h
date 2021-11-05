#ifndef  __CUDA_RENDER_H_
#define  __CUDA_RENDER_H_
class Vertex;
class Triangle;

#define M_PI 3.14159265359f
#define width 512	// screenwidth
#define height 512	// screenheight
#define samps  1	// samples per pixel per pass

extern float3* accumulatebuffer;
extern unsigned int vbo;
extern float* dev_triangle_p; // the cuda device pointer that points to the uploaded scene_triangles
// output buffer
extern float3* dptr;




extern int total_number_of_scene_triangles;
// scene bounding box
extern float3 scene_aabbox_min;
extern float3 scene_aabbox_max;

void disp();

void initCUDAmemoryTriMesh();
void pre_render_kernel(float3* output, float3* accumbuffer, const int numscene_triangles, int framenumber, int hashedframenumber, float3 scene_bbmin, float3 scene_bbmax, 
	std::vector<float4> cuda_scene_triangles, float* cudaSlabLimit, int* cudaTreeInfo, int bvh_node_num);

#endif // ! __CUDA_RENDER_H_

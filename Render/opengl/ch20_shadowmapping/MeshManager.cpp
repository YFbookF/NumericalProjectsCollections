#include "MeshManager.h"

MeshMananger::MeshMananger()
{
}

MeshMananger::~MeshMananger()
{
}

void MeshMananger::RenderCube()
{
	// initialize (if necessary)
	if (cubeVAO == 0)
	{
		float z_offset = 0;

		float vertices[] = {
			// back face
			-1.0f, -1.0f, -1.0f  ,  0.0f,  0.0f, 0.5f, 0.0f, 0.0f, // bottom-left
			 1.0f,  1.0f, -1.0f  , 0.0f,  0.0f, 0.5f, 1.0f, 1.0f, // top-right
			 1.0f, -1.0f, -1.0f  ,  0.0f,  0.0f, 0.5f, 1.0f, 0.0f, // bottom-right         
			 1.0f,  1.0f, -1.0f  ,  0.0f,  0.0f, 0.5f, 1.0f, 1.0f, // top-right
			-1.0f, 1.0f, -1.0f  ,  0.0f,  0.0f, 0.5f, 0.0f, 0.0f, // bottom-left
			-1.0f,  -1.0f, -1.0f  ,  0.0f,  0.0f, 0.5f, 0.0f, 1.0f, // top-left
			
			// front face
			-1.0f, -1.0f,  1.0f  ,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			 1.0f, -1.0f,  1.0f  ,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
			 1.0f,  1.0f,  1.0f  ,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			 1.0f,  1.0f,  1.0f  ,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
			-1.0f,  1.0f,  1.0f  ,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
			-1.0f, -1.0f,  1.0f  ,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
			
			// left face
			-1.0f,  1.0f,  1.0f  , 0.5f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			-1.0f,  1.0f, -1.0f  , 0.5f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
			-1.0f, -1.0f, -1.0f  ,0.5f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f, -1.0f  , 0.5f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
			-1.0f, -1.0f,  1.0f  , 0.5f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f,  1.0f,  1.0f  , 0.5f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
			// right face
			 1.0f,  1.0f,  1.0f  ,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			 1.0f, -1.0f, -1.0f  ,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			 1.0f,  1.0f, -1.0f  ,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
			 1.0f, -1.0f, -1.0f  ,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
			 1.0f,  1.0f,  1.0f  ,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
			 1.0f, -1.0f,  1.0f  ,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
			// bottom face
			-1.0f, -1.0f, -1.0f  ,  0.0f, 0.5f,  0.0f, 0.0f, 1.0f, // top-right
			 1.0f, -1.0f, -1.0f  ,  0.0f, 0.5f,  0.0f, 1.0f, 1.0f, // top-left
			 1.0f, -1.0f,  1.0f  ,  0.0f, 0.5f,  0.0f, 1.0f, 0.0f, // bottom-left
			 1.0f, -1.0f,  1.0f  ,  0.0f, 0.5f,  0.0f, 1.0f, 0.0f, // bottom-left
			-1.0f, -1.0f,  1.0f  ,  0.0f, 0.5f,  0.0f, 0.0f, 0.0f, // bottom-right
			-1.0f, -1.0f, -1.0f  ,  0.0f, 0.5f,  0.0f, 0.0f, 1.0f, // top-right
			// top face
			-1.0f,  1.0f, -1.0f  ,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			 1.0f,  1.0f , 1.0f  ,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			 1.0f,  1.0f, -1.0f  ,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
			 1.0f,  1.0f,  1.0f  ,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
			-1.0f,  1.0f, -1.0f  ,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
			-1.0f,  1.0f,  1.0f  , 0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
		};

		glGenVertexArrays(1, &cubeVAO);
		glGenBuffers(1, &cubeVBO);
		// fill buffer
		glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		// link vertex attributes
		glBindVertexArray(cubeVAO);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
	// render Cube
	glBindVertexArray(cubeVAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);

}

void MeshMananger::RenderCube(float start_x, float start_y, float start_z, float length_x, float length_y, float length_z)
{
	// initialize (if necessary)
	float end_x = start_x + length_x;
	float end_y = start_y + length_y;
	float end_z = start_z + length_z;

	if (planeVAO == 0)
	{

		float vertices[] = {
			// back face
			start_x, start_y, start_z, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,       /* bottom-left */
			end_x, end_y, start_z, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,           /* top-right */
			end_x, start_y, start_z, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f,         /* bottom-right */
			end_x, end_y, start_z, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,           /* top-right */
			start_x, start_y, start_z, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,       /* bottom-left */
			start_x, end_y, start_z, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f,         /* top-left */
			/* front face */
			start_x, start_y, end_z, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,          /* bottom-left */
			end_x, start_y, end_z, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,            /* bottom-right */
			end_x, end_y, end_z, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,              /* top-right */
			end_x, end_y, end_z, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,              /* top-right */
			start_x, end_y, end_z, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,            /* top-left */
			start_x, start_y, end_z, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,          /* bottom-left */
			/* left face */
			start_x, end_y, end_z, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,           /* top-right */
			start_x, end_y, start_z, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f,         /* top-left */
			start_x, start_y, start_z, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,       /* bottom-left */
			start_x, start_y, start_z, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,       /* bottom-left */
			start_x, start_y, end_z, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,         /* bottom-right */
			start_x, end_y, end_z, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,           /* top-right */
			/* right face */
			end_x, end_y, end_z, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,              /* top-left */
			end_x, start_y, start_z, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,          /* bottom-right */
			end_x, end_y, start_z, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,            /* top-right */
			end_x, start_y, start_z, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,          /* bottom-right */
			end_x, end_y, end_z, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,              /* top-left */
			end_x, start_y, end_z, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,            /* bottom-left */
			/* bottom face */
			start_x, start_y, start_z, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,       /* top-right */
			end_x, start_y, start_z, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f,         /* top-left */
			end_x, start_y, end_z, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,           /* bottom-left */
			end_x, start_y, end_z, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,           /* bottom-left */
			start_x, start_y, end_z, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,         /* bottom-right */
			start_x, start_y, start_z, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,       /* top-right */
			/* top face */
			start_x, end_y, start_z, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,          /* top-left */
			end_x, end_y, end_z, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,              /* bottom-right */
			end_x, end_y, start_z, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,            /* top-right */
			end_x, end_y, end_z, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,              /* bottom-right */
			start_x, end_y, start_z, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,          /* top-left */
			start_x, end_y, end_z, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f             /* bottom-left */
		};


		glGenVertexArrays(1, &planeVAO);
		glGenBuffers(1, &planeVBO);
		// fill buffer
		glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
		// link vertex attributes
		glBindVertexArray(planeVAO);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
	// render Cube
	glBindVertexArray(planeVAO);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glBindVertexArray(0);

}
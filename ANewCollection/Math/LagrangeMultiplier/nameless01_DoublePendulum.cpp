#include "DoublePendulum.h"
https://github.com/zetan/ConstrainedDynamics
DoublePendulum::DoublePendulum(){
	Particle A;
	A.setMass(1);
	A.setPos(Vector3D(0, 0, 0));
	A.setVelocity(Vector3D(0, 0, 0));
	A.setForce(Vector3D(0, 0, 0));
	float radius = 1;
	Particle B;
	B.setMass(1);
	B.setPos(Vector3D(radius, 0, 0));
	B.setVelocity(Vector3D(0, 0, 0));
	B.setForce(Vector3D(0, 0, 0));

	particles.push_back(A);
	particles.push_back(B);

	//init constrain force parameters
	constrainForce.setAlpha(1);
	constrainForce.setRadius(1);
} 

void DoublePendulum::Init(){
	particles[0].setMass(1);
	particles[0].setPos(Vector3D(0, 0, 0));
	particles[0].setVelocity(Vector3D(0, 0, 0));
	particles[0].setForce(Vector3D(0, 0, 0));
	particles[1].setMass(1);
	particles[1].setPos(Vector3D(1, 0, 0));
	particles[1].setVelocity(Vector3D(0, 0, 0));
	particles[1].setForce(Vector3D(0, 0, 0));
	spring1.setActive(false);
	spring2.setActive(false);
}

void DoublePendulum::ApplyForces(){
	ClearForces();
	//qDebug()<<(constrainForce.getAlpha() * particles[0].getPos().x * particles[0].getPos().x - particles[0].getPos().y);
	//qDebug()<<particles[0].getPos().y;
	
	//energy
	double energy = 0;
	for(int i = 0; i < particles.size(); i++){
		energy += 10 * particles[i].getPos().y;
		energy += Vector3D::DotProduct(particles[i].getVelocity(), particles[i].getVelocity()) / 2;
	}
	qDebug()<<energy;
	for(int i = 0; i < particles.size(); i++){
		gravity.ApplyForce(&particles[i]);
	}
	spring1.ApplyForce(&particles[0]);
	spring2.ApplyForce(&particles[1]);
	constrainForce.ApplyForce(particles);
}

void DoublePendulum::Draw(){
	glColor3f(1.0, 1.0, 1.0);
	DrawParabola();
	for(int i = 0; i < particles.size(); i++)
		particles[i].Draw();
	
	DrawLine();

	glColor3f(0.0, 1.0, 1.0);
	DrawSpring();

	glColor3f(1.0, 0.0, 0.0);
	GLDraw::DrawArrow(particles[0].getPos(), Vector3D::Scale(constrainForce.getConstrainForceA(), 0.05));
	GLDraw::DrawArrow(particles[1].getPos(), Vector3D::Scale(constrainForce.getConstrainForceB(), 0.05));
}

void DoublePendulum::DrawParabola(){
	double alpha = constrainForce.getAlpha();
	glBegin( GL_LINE_STRIP_ADJACENCY );
	for (double x = -1.3; x <= 1.31; x+= 0.05)
	{
	//	float degInRad = i * 3.1415/ 180;
		glVertex3f(x,alpha*x*x, 0);
	}
	glEnd();
}

void DoublePendulum::DrawLine(){
	glBegin( GL_LINE_LOOP );
	glVertex3f(particles[0].getPos().x, particles[0].getPos().y, 0);
	glVertex3f(particles[1].getPos().x, particles[1].getPos().y, 0);
	glEnd();
}

void DoublePendulum::setMouseSpring(Vector3D start, Vector3D end){
	double len1 = Vector3D::Minus(start, particles[0].getPos()).Length();
	double len2 = Vector3D::Minus(start, particles[1].getPos()).Length();
	if(len1 < len2){
		spring1.setPt(end);
		spring1.setRestLength(Vector3D::Minus(end, particles[0].getPos()).Length());
		spring1.setActive(true);
	}
	else{
		spring2.setPt(end);
		spring2.setRestLength(Vector3D::Minus(end, particles[1].getPos()).Length());
		spring2.setActive(true);
	}
}

void DoublePendulum::DrawSpring(){
	if(spring1.isActive() == true){
		glBegin( GL_LINE_LOOP );
		glVertex3f(particles[0].getPos().x, particles[0].getPos().y, 0);
		glVertex3f(spring1.getPt().x, spring1.getPt().y, 0);
		glEnd();
		glTranslatef( spring1.getPt().x, spring1.getPt().y, 0);
		GLDraw::DrawCircle(0.02);

		glTranslatef(-spring1.getPt().x, -spring1.getPt().y, 0);
	}
	if(spring2.isActive() == true){
		glBegin( GL_LINE_LOOP );
		glVertex3f(particles[1].getPos().x, particles[1].getPos().y, 0);
		glVertex3f(spring2.getPt().x, spring2.getPt().y, 0);
		glEnd();
		glTranslatef(spring2.getPt().x, spring2.getPt().y, 0);
		GLDraw::DrawCircle(0.02);
		glTranslatef(-spring2.getPt().x, -spring2.getPt().y, 0);
	}
}
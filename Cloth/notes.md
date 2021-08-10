那个阻尼公式

```
virtual void ApplyForce()
	{
		const real kStiffness = 10000.0f;
		const real kDamping = 50.0f;
		
		// position delta
		Vec2r dx = mP->mX - mQ->mX;
		real dl = Length(dx);
		dx /= dl;
		
		// velocity delta
		Vec2r dv = mP->mV - mQ->mV;
		
		Vec2r f = (kStiffness*(dl-mRestLength) + kDamping*(Dot(dv, dx)))*dx;
		
		mP->ApplyForce(-f);
		mQ->ApplyForce(f);
	}
	
```


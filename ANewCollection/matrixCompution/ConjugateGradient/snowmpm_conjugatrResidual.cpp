https://github.com/Azmisov/snow
	//LINEAR SOLVE
	for (int i=0; i<MAX_IMPLICIT_ITERS; i++){
		bool done = true;
		for (int idx=0; idx<nodes_length; idx++){
			GridNode& n = nodes[idx];
			//Only perform calculations on nodes that haven't been solved yet
			if (n.imp_active){
				//Alright, so we'll handle each node's solve separately
				//First thing to do is update our vf guess
				float div = n.Ep.dot(n.Ep);
				float alpha = n.rEr / div;
				n.err = alpha*n.p;
				//If the error is small enough, we're done
				float err = n.err.length();
				if (err < MAX_IMPLICIT_ERR || err > MIN_IMPLICIT_ERR || isnan(err)){
					n.imp_active = false;
					continue;
				}
				else done = false;
				//Update vf and residual
				n.velocity_new += n.err;
				n.r -= alpha*n.Ep;
			}
		}
		//If all the velocities converged, we're done
		if (done) break;
		//Otherwise we recompute Er, so we can compute our next guess
		recomputeImplicitForces();
		//Calculate the gradient for our next guess
		for (int idx=0; idx<nodes_length; idx++){
			GridNode& n = nodes[idx];
			if (n.imp_active){
				float temp = n.r.dot(n.Er);
				float beta = temp / n.rEr;
				n.rEr = temp;
				//Update p
				n.p *= beta;
				n.p += n.r;
				//Update Ep
				n.Ep *= beta;
				n.Ep += n.Er;
			}
		}
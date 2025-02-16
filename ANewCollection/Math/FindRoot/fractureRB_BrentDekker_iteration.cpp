https://github.com/david-hahn/FractureRB
        // Brent-Dekker iteration
		// solve approximation problem f(x) = 0
		//step 1. if |b - c| < delta , the method returns b as the approximation solution
        while( found && std::abs(fa) > eps && std::abs(b-a)>eps ) {
            fc= evalKthMinusKcGradient(
                Kth, dKth_dth, Kc_p, dKc_dn, dKc,
                c, K, pos, n1, n2
            );
		// step 2. if a = c then b is determined by 
		//linear (secant) interpolations b = (af(b) - bf(a))/(f(b) - f(a))
            if( std::abs(fa-fc)>eps && std::abs(fb-fc)>eps ){ // can use inv quad interp
                s = a*fb*fc/((fa-fb)*(fa-fc)) + b*fa*fc/((fb-fa)*(fb-fc)) + c*fa*fb/((fc-fa)*(fc-fb));
            }else{ // use lin interp
                s = b - fb*(b-a)/(fb-fa);
            }
            if( (s < 0.25*(3.0*a+b) || s > b) ||
                (mflag==true  && std::abs(s-b) >= 0.5*std::abs(b-c)) ||
                (mflag==false && std::abs(s-b) >= 0.5*std::abs(c-d)) ||
                (mflag==true  && std::abs(b-c) < eps) ||
                (mflag==false && std::abs(c-d) < eps)
            ){
                s = 0.5*(a+b); mflag=true;
            }else mflag=false;
            fs= evalKthMinusKcGradient(
                Kth, dKth_dth, Kc_p, dKc_dn, dKc,
                s, K, pos, n1, n2
            );
            d=c; c=b; fc=fb;
            if( fa*fs < 0.0){ b=s; fb=fs; }
            else { a=s; fa=fs; }
            if(std::abs(fa)<std::abs(fb)){ //swap (a,b)
                tmp = a; a = b; b = tmp;
                tmp =fa; fa=fb; fb= tmp;
            }
        }
		if(found){
            // theta must always be in [-pi/2,+pi/2]
            th = asin(sin(a));
        }
        return th;
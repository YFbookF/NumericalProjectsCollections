https://github.com/matthias-research/pages/blob/master/challenges/pendulum.html
<!-- Pendulum Simulator -->
<!-- Matthias Müller, nvidia -->

<!DOCTYPE html>
<html>
<head>
<style>
th, td {
	padding: 2px;
}
body {
	padding: 10px 50px;
	font-family: verdana; 
	line-height: 1.5;
	font-size: 15px;
}
h1 {
	font-family: verdana; 
}
#gui {
	padding: 10px;
}
.button {
  background-color: #555555;
  border: none;
  color: white;
  padding: 8px 8px;
  border-radius: 5px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
}
.slider {
  -webkit-appearance: none;
  width: 80px;
  height: 6px;
  border-radius: 5px;
  background: #d3d3d3;
  outline: none;
  opacity: 0.7;
  -webkit-transition: .2s;
  transition: opacity .2s;
}
.slider:hover {
  opacity: 1;
}
.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  background: #202020;
  cursor: pointer;
}
</style>
<script src="https://www.powr.io/powr.js?platform=embed"></script>
</head>


<title>The Pendulum Challenge</title>
<body>

<h1>The Pendulum Challenge</h1>
Matthias M&uuml;ller, Nvidia
<br>
<br>
<h3 style = "color: #B93226;">This simulation shows that extended position based dynamics (XPBD) is a strong competitor to state of the art simulation methods in terms of accuracy, stability, speed and simplicity. To understand why and how to use the demo see the text below.</h3>.
<br>

<table>
  <tr>
  <td>
  <button onclick="resetPos(false)" class="button">Restart</button>
	<button onclick="resetPos(true)" class="button">Equilibrium position</button>
	<button onclick="step()" class="button">Step</button>
	<button onclick="run()" class="button">Run</button>
	<br><br>
<canvas id="myCanvas" width="500" height="500" style="border:3px solid #d3d3d3;">
Your browser does not support the HTML5 canvas tag.</canvas>
</td>
<td id = "gui">
<p><span id = "ms">0.000</span> ms per frame, dt = 1/60 s, g = -10 m/s<sup>2</sup></p>
<p>Number of links: <input type = "range" min = "1" max = "4" value = "3" id = "segsSlider"  class = "slider"> <span id = "numSegs">3</span></p>
<p>Number of sub-steps: <input type = "range" min = "0" max = "6" value = "4" id = "stepsSlider" class = "slider"> <span id = "steps">50</span></p>
<table>
<tr>
<th>mass (kg)</th><th>length (m)</th><th>compliance (m/N)</th><th>uni</th>
</tr>
<tr>
<td><input type = "range" min = "0" max = "4" value = "2" id = "mass1Slider" class = "slider"> <span id = "mass1">1.0</span></td>
<td><input type = "range" min = "0" max = "2" value = "1" id = "radius1Slider" class = "slider"> <span id = "radius1">0.3</span></td>
<td><input type = "range" min = "0" max = "2" value = "0" id = "compliance1Slider" class = "slider"> <span id = "compliance1">0.000</span></td>
<td><input type = "checkbox" onclick = "onUnilateral(1)"></td>
</tr>
<tr>
<td><input type = "range" min = "0" max = "4" value = "2" id = "mass2Slider" class = "slider"> <span id = "mass2">1.0</span></td>
<td><input type = "range" min = "0" max = "2" value = "1" id = "radius2Slider" class = "slider"> <span id = "radius2">0.3</span></td>
<td><input type = "range" min = "0" max = "2" value = "0" id = "compliance2Slider" class = "slider"> <span id = "compliance2">0.000</span></td>
<td><input type = "checkbox" onclick = "onUnilateral(2)"></td>
</tr>
<tr>
<td><input type = "range" min = "0" max = "4" value = "2" id = "mass3Slider"  class = "slider"> <span id = "mass3">1.0</span></td>
<td><input type = "range" min = "0" max = "2" value = "1" id = "radius3Slider" class = "slider"> <span id = "radius3">0.3</span></td>
<td><input type = "range" min = "0" max = "2" value = "0" id = "compliance3Slider" class = "slider"> <span id = "compliance3">0.000</span></td>
<td><input type = "checkbox" onclick = "onUnilateral(3)"></td>
</tr>
<tr>
<td><input type = "range" min = "0" max = "4" value = "2" id = "mass4Slider" class = "slider"> <span id = "mass4">1.0</span></td>
<td><input type = "range" min = "0" max = "2" value = "1" id = "radius4Slider" class = "slider"> <span id = "radius4">0.3</span></td>
<td><input type = "range" min = "0" max = "2" value = "0" id = "compliance4Slider" class = "slider"> <span id = "compliance4">0.000</span></td>
<td><input type = "checkbox" onclick = "onUnilateral(4)"></td>
</tr>
</table>
<p>Edge damping coefficient (Ns/m) <input type = "range" min = "0" max = "2" value = "0" id = "edgeDampingSlider" class = "slider"> <span id = "edgeDamping">0.0</span></p>
<p>Global damping coefficient (Ns/m) <input type = "range" min = "0" max = "3" value = "0" id = "globalDampingSlider" class = "slider"> <span id = "globalDamping">0.0</span></p>
<p><input type = "checkbox" onclick = "onEnergy()"> Enforce energy conservation</p>
<p><input type = "checkbox" onclick = "onCollision()"> Collision handling</p>
<p><input type = "checkbox" onclick = "onTrail()" checked> Show trail
<input type = "checkbox" onclick = "onForces()"> Force labels</p>
</td>
</table>
<p>
This pendulum simulation shows <b>single step XPBD</b> in action. Two small modifications in our original position based dynamics method <b>PBD</b> turns the latter from a toy used in games into a serious competitor of physical simulation methods even surpassing their accuracy while staying as simple as <b>PBD</b>. We describe the method <a href = "https://matthias-research.github.io/pages/publications/smallsteps.pdf" target="_blank" class="paperButton"> here</a>. The pendulum simulation showcases the following features: It handles stiff systems with large mass ratios. (Zero compliance means infinite stiffness). It shows high frequency details and a high level of energy conservation both of which are difficult to achieve with implicit global solvers. All quantities have physical units and internal forces can easily be evaluated. It removes the conceptual differences of <b>PBD</b> to the backward Euler method. It also generalizes beyond the distance constraints we show here.
<br><br>
The challenge is to <b>beat our method</b> in terms of simplicity, speed, stability or accuracy. The html document you see here is self-contained. It includes the GUI, the rendering and the complete simulation code. You can have a look at it <a href = "https://github.com/matthias-research/pages/blob/master/challenges/pendulum.html" target = "_blank">here</a>. The solver itself only takes 100 lines of code (lines 460-560) which you can simply replace it with your own solution.
<ul>
  <li><b>The triple pendulum</b><br>
  The demo starts with a triple pendulum. The reason is that there are quite a few double pendulum simulations on the web. While their reduced coordinates formulation is reasonable complex, the equations for the triple pendulum cover an entire page. Although our method conserves energy quite well, the simulation comes to a stop eventually. If you want to look at it forever, turn on "Enforce energy conservation". I do not recommend to use this feature in general however. </li>
 <li><b>Mouse interaction</b><br>
  You can use the mouse to drag the masses. The mouse pointer is attached to the weights via a spring. This allows you to experience the weight variation and the stiffness of the rods.
  </li>  
  <li><b>Number of sub-steps</b><br>
  The crucial idea to tremendously increase the convergence rate of PBD was to replace iterations by sub-steps. We have set the number of sub-steps to 50. Almost all demos work well with 20. The only reason to go higher is to reduce the amount of energy loss. Note that effect of compliance is independent of the number of sub-steps due to the XPBD update.
  </li>
  <li><b>Force labels</b><br>
  Set the equilibrium position and turn on force labels. As you can see, the forces correspond to 10 times the weight below each link since gravity is 10. At the same time, the elongations are zero if the compliance is zero (corresponding to infinite stiffness). If you increase the compliance the elongations become proportional (via inverse compliance) to the forces. This is best seen with high edge damping. Pull on the weights to see what happens.</li>
  <li>
  <b>The single pendulum</b><br>
  Set the number of segments to 1. Choose compliance zero. As expected from physics, the frequency is independent of the mass but dependent on the link length. Set the mass to ten and the compliance to 0.01. Then hit restart with a variety of edge damping coefficients. While the demo is running, change the number of sub-steps. As you see, the stiffness is unaffected. 
  </li>
  <li><b>The double pendulum</b><br>
  Compare the behavior with the many simulations on the web. Play with all the parameters.</li>
  <li><b>Unilateral constraints</b></br>
  Turn compliance to zero and check the "uni" box. The links turn green. In this case, they are allowed to compress but not to expand. The corresponding constraints are called unilateral. Handling them with traditional solvers is challenging.</li>
  <li><b>Collisions</b><br>
  Turn on collision handling and see what happens. The last weight bounces off the x = 0 line. This experiment shows the high fidelity of you method. In this case, the trajectory has sharp edges. These are damped out with implicit solvers. The problem increases by increasing the order of the integration method. Collision constraints are unilateral as well.
  </li>
  <li><b>High mass ratios</b><br>
  To see how well our method handles high mass ratios, set the number of links to four. Set the firs three masses to 0.1 and the last to 10 while all compliances are zero. Hit restart and see how there is almost no stretching in the links. Play with the number of sub-steps. Twenty are enough to handle this situation.
  </li>
  <li><b>Damping</b><br>
  Edge damping decreases the bouncing of compliant links. As stiffness, damping is unconditionally stable with our method. Sometimes, users like the overall damping introduced artificially be implicit solvers. You can inject it in a controlled manner via the global damping parameter.</li>  
  <li><b>Stability</b></br>
  Note the stability of our approach over the wide space of parameters. We have not seen serious crashes but if it happens, simply re-load the page by hitting F5.
</ul> 
</p>
<a href = "https://matthias-research.github.io/pages/challenges/challenges.html" class="button">More challenges</a>
<br>
<div class="powr-comments" id="968e209d_1569590391"></div>

<script>

	// global parameters
	
	var numSubsteps = 50;
	var numPoints = 4;
	var defaultRadius = 0.3;
	var defaultMass = 1.0;
	var gravity = 10;
	var dt = 1 / 60;
	var edgeDampingCoeff = 0;
	var globalDampingCoeff = 0;
	
	var conserveEnergy = false;
	var collisionHandling = false;
	var showTrail = true;
	var showForces = false;
	var maxPoints = 5;
	
	var maxTrailLen = 1000;
	var trailDist = 0.01;
	
	var mouseCompliance = 0.001;
	var mouseDampingCoeff = 100.0;
	
	var canvas = document.getElementById("myCanvas");
	var c = canvas.getContext("2d");
	var canvasOrig = { x : canvas.width / 2, y : canvas.height / 4};
	var simWidth = 2.0;
	var pointSize = 10;
	var drawScale = canvas.width / simWidth;
	
	var i,j;
	
	// GUI callbacks
	
	document.getElementById("stepsSlider").oninput = function() {
		var steps = [1, 5, 10, 20, 50, 100, 1000];
		numSubsteps = steps[Number(this.value)];
		document.getElementById("steps").innerHTML = numSubsteps.toString();
	}
	document.getElementById("segsSlider").oninput = function() {
		numPoints = Number(this.value) + 1;
		document.getElementById("numSegs").innerHTML = this.value;
		resetPos(false);
	}

	document.getElementById("edgeDampingSlider").oninput = function() {
		var coeffs = ["0.0", "10.0", "100.0"];
		var coeff = coeffs[Number(this.value)];
		edgeDampingCoeff = Number(coeff);		
		document.getElementById("edgeDamping").innerHTML = coeff;
	}

	document.getElementById("globalDampingSlider").oninput = function() {
		var coeffs = ["0.0", "0.5", "1.0", "2.0"];
		var coeff = coeffs[Number(this.value)];
		globalDampingCoeff = Number(coeff);		
		document.getElementById("globalDamping").innerHTML = coeff;
	}
	 
	function setupMass(value, output, pointNr) {
		var masses = ["0.001", "0.5", "1.0", "2.0", "10"];
		var m = masses[value];
		document.getElementById(output).innerHTML = m;
		points[pointNr].invMass = 1.0 / Number(m);
		points[pointNr].size = Math.sqrt(Number(m));
	}
	
	function setupRadius(value, output, pointNr) {
		var lengths = ["0.2", "0.3", "0.4"];
		var len = lengths[value];
		document.getElementById(output).innerHTML = len;
		points[pointNr].radius = Number(len);
		resetPos(false);
	}

	function setupCompliance(value, output, pointNr) {
		var values = ["0.000", "0.001", "0.010"];
		var compliance = values[value];
		document.getElementById(output).innerHTML = compliance;
		points[pointNr].compliance = Number(compliance);
	}
	
	document.getElementById("mass1Slider").oninput = function() {
		setupMass(Number(this.value), "mass1", 1);
	}
	document.getElementById("mass2Slider").oninput = function() {
		setupMass(Number(this.value), "mass2", 2);
	}
	document.getElementById("mass3Slider").oninput = function() {
		setupMass(Number(this.value), "mass3", 3);
	}
	document.getElementById("mass4Slider").oninput = function() {
		setupMass(Number(this.value), "mass4", 4);
	}
	document.getElementById("radius1Slider").oninput = function() {
		setupRadius(Number(this.value), "radius1", 1);
	}
	document.getElementById("radius2Slider").oninput = function() {
		setupRadius(Number(this.value), "radius2", 2);
	}
	document.getElementById("radius3Slider").oninput = function() {
		setupRadius(Number(this.value), "radius3", 3);
	}
	document.getElementById("radius4Slider").oninput = function() {
		setupRadius(Number(this.value), "radius4", 4);
	}
	document.getElementById("compliance1Slider").oninput = function() {
		setupCompliance(Number(this.value), "compliance1", 1);
	}
	document.getElementById("compliance2Slider").oninput = function() {
		setupCompliance(Number(this.value), "compliance2", 2);
	}
	document.getElementById("compliance3Slider").oninput = function() {
		setupCompliance(Number(this.value), "compliance3", 3);
	}
	document.getElementById("compliance4Slider").oninput = function() {
		setupCompliance(Number(this.value), "compliance4", 4);
	}
		
	function onEnergy() {
		conserveEnergy = !conserveEnergy;
		resetPos(false);
	}
	
	function onCollision() {
		collisionHandling = !collisionHandling;
		resetPos(false);
	}

	function onTrail() {
		showTrail = !showTrail;
		trail = [];
		trailLast = 0;
	}

	function onForces() {
		showForces = !showForces;
	}
	
	function onUnilateral(nr) {
		points[nr].unilateral = !points[nr].unilateral;
	}
		
	class Vector {
		constructor(x = 0, y = 0) { this.x = x; this.y = y; }
		copy(v)   { 
			return new Vector(this.x, this.y); 
		}
		assign(v) { 
			this.x = v.x; this.y = v.y; 
		}
		plus(v) { 
			return new Vector(this.x + v.x, this.y + v.y); 
		}
		minus(v) { 
			return new Vector(this.x - v.x, this.y - v.y); 
		}
		add(v, s = 1) { 
			this.x += v.x * s; this.y += v.y * s; 
		}
		scale(s) {
			this.x *= s; this.y *= s; 
		}
		dot(v) { 
			return this.x * v.x + this.y * v.y; 
		}
		normalize() {
			var d = Math.sqrt(this.x * this.x + this.y * this.y);		
			if (d > 0) { this.x /= d; this.y /= d; } else this.x = 1;
			return d;
		}
		lenSquared() { 
			return this.x * this.x + this.y * this.y; 
		}
		distSquared(v) { 
			return (this.x - v.x) * (this.x - v.x) + (this.y - v.y) * (this.y - v.y);
		}
	}
			
	// trail
	
	var trailLast = 0;
	var trail = [];

	function trailAdd(p) {
		if (trail.length == 0)
			trail.push(p.copy());
		else {
			var d2 = trail[trailLast].distSquared(p);
			if (d2 > trailDist * trailDist) {
				trailLast = (trailLast + 1) % maxTrailLen;
				if (trail.length < maxTrailLen)
					trail.push(p.copy());
				else 
					trail[trailLast].assign(p);
			}
		}
	}
	
	// pendulum definition
	
	var points = [];
	for (i = 0; i < maxPoints; i++)
		points.push(
		{
			invMass: i == 0 ? 0 : 1 / defaultMass,
			radius: i == 0 ? 0 : defaultRadius,
			size: 0,
			pos: new Vector(),
			prev: new Vector(),
			vel: new Vector(), 
			compliance : 0,
			unilateral : false,
			force : 0,
			elongation : 0,
		});
					
	function resetPos(equilibrium) 
	{
		var pos = equilibrium ? new Vector(0, 0) : new Vector(points[1].radius, - points[1].radius);

		for (i = 1; i < points.length; i++) {
			p = points[i];
			p.size = Math.sqrt(1.0 / p.invMass);
			pos.y = equilibrium ? pos.y - p.radius : pos.y + p.radius;
			p.pos.assign(pos); p.prev.assign(pos); 
			p.vel.x = 0; p.vel.y = 0;
		}		
		trail = [];
		trailLast = 0;
		draw();
	}
	
	// draw pendulum
		
	function draw() {
		c.clearRect(0, 0, canvas.width, canvas.height);

		c.lineWidth = 3;
		c.font = "15px Arial";

		var x = canvasOrig.x;
		var y = canvasOrig.y;

		for (i = 1; i < numPoints; i++) {
			var avgX = x, avgY = y;
			p = points[i];
			if (p.compliance > 0) c.strokeStyle = "#0000FF";
			else if (p.unilateral) c.strokeStyle = "#00FF00";
			else c.strokeStyle = "#000000";
			c.beginPath();
			c.moveTo(x, y);
			x = canvasOrig.x + p.pos.x * drawScale;
			y = canvasOrig.y - p.pos.y * drawScale;
			c.lineTo(x, y);
			c.stroke();
			avgX = (avgX + x) / 2; avgY = (avgY + y) / 2;

			if (showForces)			
				c.fillText("  f=" + p.force.toFixed(0) + "N, dx=" + p.elongation.toFixed(4) + "m", avgX, avgY);
			
		}
		c.lineWidth = 1;
		
		if (grabPointNr > 0) {
			c.strokeStyle = "#FF8000";
			c.beginPath();
			c.moveTo(canvasOrig.x + grabPoint.pos.x * drawScale, canvasOrig.y - grabPoint.pos.y * drawScale);
			c.lineTo(canvasOrig.x + points[grabPointNr].pos.x * drawScale, canvasOrig.y - points[grabPointNr].pos.y * drawScale);
			c.stroke();
		}
		
		for (i = 1; i < numPoints; i++) {
			p = points[i];
			x = canvasOrig.x + p.pos.x * drawScale;
			y = canvasOrig.y - p.pos.y * drawScale;
			c.beginPath();			
			c.arc(x, y, pointSize * p.size, 0, Math.PI*2, true); 
			c.closePath();
			c.fill();			
		}		
		
		if (trail.length > 1) {
			c.strokeStyle = "#FF0000";
			c.beginPath();
			var pos = (trailLast + 1) % trail.length;
			c.moveTo(canvasOrig.x + trail[pos].x * drawScale, canvasOrig.y - trail[pos].y * drawScale);
			for (i = 0; i < trail.length - 1; i++) {
				pos = (pos + 1) % trail.length;
				c.lineTo(canvasOrig.x + trail[pos].x * drawScale, canvasOrig.y - trail[pos].y * drawScale);
			}
			c.stroke();
			c.strokeStyle = "#000000";
		}
    }
		
    // simulation (replace with yours) ------------------------------------------------------------
	
	function solveDistPos(p0, p1, d0, compliance, unilateral, dt) 
	{
		var w = p0.invMass + p1.invMass;
		if (w == 0)
			return;
		var grad = p1.pos.minus(p0.pos);
		var d = grad.normalize();
		w += compliance / dt / dt;
		var lambda = (d - d0) / w;
				
		if (lambda < 0 && unilateral)
			return;
		p1.force = lambda / dt / dt;
		p1.elongation = d - d0;
		p0.pos.add(grad, p0.invMass * lambda);
		p1.pos.add(grad, -p1.invMass * lambda);
	}

	function solveDistVel(p0, p1, dampingCoeff, dt) 
	{
		var n = p1.pos.minus(p0.pos);
		n.normalize();
		var v0 = n.dot(p0.vel);
		var v1 = n.dot(p1.vel);		
		var dv0 = (v1 - v0) * Math.min(0.5, dampingCoeff * dt * p0.invMass);
		var dv1 = (v0 - v1) * Math.min(0.5, dampingCoeff * dt * p1.invMass);
		p0.vel.add(n, dv0);
		p1.vel.add(n, dv1);
	}

	function solvePointVel(p, dampingCoeff, dt) 
	{
		var n = p.vel.copy()
		var v = n.normalize();
		var dv = -v * Math.min(1.0, dampingCoeff * dt * p.invMass);
		p.vel.add(n, dv);
	}
	
	function simulate(dt) 
	{
		var sdt = dt / numSubsteps;
		var step;
		for (step = 0; step < numSubsteps; step++) {

			// predict

			for (i = 1; i < numPoints; i++) {
				p = points[i];
				p.vel.y -= gravity * sdt;
				p.prev.assign(p.pos);
				p.pos.add(p.vel, sdt);
			}

			// solve positions

			for (i = 0; i < numPoints - 1; i++) {
				p = points[i + 1];
				solveDistPos(points[i], p, p.radius, p.compliance, p.unilateral, sdt);
			}

			if (grabPointNr >= 0) 
				solveDistPos(grabPoint, points[grabPointNr], 0, mouseCompliance, false, sdt);

			if (collisionHandling) {
				var minX = 0;
				p = points[numPoints - 1];
				if (p.pos.x < minX) {
					p.pos.x = minX;
					if (p.vel.x < 0)
						p.prev.x = p.pos.x + p.vel.x * sdt;
				}
			}
			
			// update velocities

			for (i = 1; i < numPoints; i++) {
				p = points[i];
				p.vel = p.pos.minus(p.prev);
				p.vel.scale(1 / sdt);
				solvePointVel(p, globalDampingCoeff, sdt);
			}
			
			for (i = 0; i < numPoints - 1; i++) {
				p = points[i + 1];
				if (p.compliance > 0.0)
					solveDistVel(points[i], p, edgeDampingCoeff, sdt);
			}
			if (grabPointNr >= 0) 
				solveDistVel(grabPoint, points[grabPointNr], mouseDampingCoeff, sdt);
			
			if (showTrail)
				trailAdd(points[numPoints-1].pos);
		}
	}
	
	// ---------------------------------------------------------------------------------------
	
	// energy conservation
	
	function computeEnergy()
	{
		var E = 0;
		for (i = 1; i < numPoints; i++) {
			p = points[i];
			E += p.pos.y / p.invMass * gravity + 0.5 / p.invMass * p.vel.lenSquared();
		}
		return E;
  	}
	
	function forceEnergyConservation(prevE)
	{
		var dE = (computeEnergy() - prevE) / (numPoints - 1);
		if (dE < 0) {
			var postE = computeEnergy();

			for (i = 1; i < numPoints; i++) {
				p = points[i];
				var Ek = 0.5 / p.invMass * p.vel.lenSquared();
				var s = Math.sqrt((Ek - dE) / Ek);
				p.vel.scale(s);
			}
		} 		
	}

	// animation
	
	var requestAnimationFrame = window.requestAnimationFrame ||
		window.mozRequestAnimationFrame ||
		window.webkitRequestAnimationFrame ||
		window.msRequestAnimationFrame;

	var timeFrames = 0;
	var timeSum = 0;
	var paused = false;

	function timeStep() 
	{
		var prevE;
		if (conserveEnergy)
			prevE = computeEnergy();	
		var startTime = performance.now();
		
		simulate(dt);

		var endTime = performance.now();
		if (conserveEnergy)
			forceEnergyConservation(prevE);
		
		timeSum += endTime - startTime; 
		timeFrames++;
		
		if (timeFrames > 10) {
			timeSum /= timeFrames;
			document.getElementById("ms").innerHTML = timeSum.toFixed(3);		
			timeFrames = 0;
			timeSum = 0;
		}
						
		draw();
		if (!paused)
			requestAnimationFrame(timeStep);
	}
	
	function step()
	{
		paused = true;
		timeStep();	
	}

	function run()
	{
		if (paused) {
			paused = false;
			timeStep();
		}
	}

	// mouse grab
	
	var grabPointNr = -1;
	var grabPoint = { pos : new Vector(), invMass : 0, vel : new Vector() };
	var maxGrabDist = 0.5;
	var prevConserveEnergy = conserveEnergy;
		
	function onMouse(evt) {
		evt.preventDefault();
		var rect = canvas.getBoundingClientRect();	
		var mousePos = new Vector(
			((evt.clientX - rect.left) - canvasOrig.x) / drawScale, 
			(canvasOrig.y - (evt.clientY - rect.top)) / drawScale);
		if (evt.type == "mousedown") {
			grabPointNr = -1;
			var minGrabDist2 = maxGrabDist * maxGrabDist;
			for (i = 1; i < numPoints; i++) {
				p = points[i];
				var d2 = p.pos.distSquared(mousePos);
				if (d2 < minGrabDist2) {
					minGrabDist2 = d2;
					grabPointNr = i;
					grabPoint.pos.assign(mousePos);
					prevConserveEnergy = conserveEnergy;
					conserveEnergy = false;
				}
			}
		}
		else if (evt.type == "mousemove") {
			grabPoint.pos.assign(mousePos);
		}
		
		else if (evt.type == "mouseup" || evt.type == "mouseout") {
			grabPointNr = -1;
			conserveEnergy = prevConserveEnergy;
		}
	}
	
	canvas.addEventListener("mousedown", onMouse);
	canvas.addEventListener("mousemove", onMouse);
	canvas.addEventListener("mouseup", onMouse);
	canvas.addEventListener("mouseout", onMouse);
	
	// main
	
	resetPos(false);
	timeStep();

</script> 
</body>
</html>
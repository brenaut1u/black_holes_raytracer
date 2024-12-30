# 3D Rendering of black holes using raytracing

Katia Brenaut, Andy Pan, Wenhao Shen

McGill University

## Introduction
Black holes have been fascinating astrophysicists for decades. We currently know quite a lot about their extraordinary properties, yet for now we only have a few very blurry pictures of them. So how do we know what they look like? This is where the physical simulations come to the rescue. Black holes simulations have many different applications, from scientific research to educational purposes, and even movies such as Interstellar, for which specialists worked hard to produce both beautiful and physically accurate results.

The purpose of this project was to develop a 3D N-body simulation program that renders an animation of several black-holes interacting with each other through gravity, using a common computer graphics method called ray tracing.

## Methodology
Ray tracing is a method used to take a virtual 2D picture of a 3D scene. It consists in setting up a virtual camera with some physical properties such as a position, a screen and a focal length. For each pixel of the screen, we launch a light ray and see where it intersects objects in the scene. The properties of the intersection point determine the color of the pixel.

Normally, the rays are straight lines, and we only need to compute intersections using these lines’ equations. However, in the context of black holes, light rays are significantly deviated. Therefore, we needed a different approach: for each pixel, we launch a photon, and iteratively move it in the scene until it either falls into a singularity, or travels beyond a certain distance to the camera.

For this project, we alternated between two steps: rendering the scene, and updating the black holes’ state according to their interactions with each other. We will go into further details later in this report.

For the simulation of the interactions between the black holes, since the objects studied have masses large enough to curve spacetime significantly, the leading-order term in the post-Newtonian expansion is used to calculate the objects’ acceleration.

The acceleration of object 𝑎 in the scene is calculated based on the leading-order term in the post-Newtonian expansion. [1]

![Equation](./Readme%20equations/eq1.png)

where:

![Equation](./Readme%20equations/eq2.png)

To avoid numerical instability when the two objects get very close, a softening parameter 𝜀 is added to the distance between two objects, so

![Equation](./Readme%20equations/eq3.png)

where 𝑠𝑐𝑎𝑙𝑒 is the length scale of our dynamical system (1012 metres). It is noted that 𝜀=5×10−2 and 𝜀=5×10−5 for objects and photons, respectively, works well to ensure numerical stability.

The positions and velocities of objects and photons are updated through basic kinematics equations (Taylor expansions of position and velocity).

![Equation](./Readme%20equations/eq4.png)

where 𝒙(𝑡+𝛿𝑡) is the updated position at the next time step, 𝒙(𝑡) and 𝒗(𝑡) are the positions and velocities at the current time step, and 𝒂(𝑡) is the acceleration based on the positions and velocities at the current time step, and 𝒋(𝑡) is the total jerk. The velocities are predicted by the equation below

![Equation](./Readme%20equations/eq5.png)

The physical quantity “jerk” is the 3rd time-derivative of position (2nd derivative of velocity and 1st derivative of acceleration). It is used to obtain a 3rd-order Taylor expansion of position and a 2nd order Taylor expansion of velocity. The total jerk of object 𝑎 is computed by adding the post-Newtonian (PN) part of the total jerk (a backward numerical derivative of the PN part of acceleration) to the Newtonian part (analytical derivative of the Newtonian part of acceleration).

![Equation](./Readme%20equations/eq6.png)

Note:

![Equation](./Readme%20equations/eq7.png)

so

![Equation](./Readme%20equations/eq8.png)

The program starts by initializing the positions and velocities of objects in the scene and consists of two steps for each frame of the animation:

Step A: the program freezes the objects in the scene, launches a photon from each screen pixel in the camera, and iteratively updates its position and velocity over several time steps 𝛿𝑡′ (𝛿𝑡′≠𝛿𝑡) based on its Newtonian acceleration until it has travelled a considerable distance (in which case it serves to display some background texture) or got below a certain distance to singularity (in which case the pixel will be black).

Step B: the program advances the objects in the scene by a timestep and updates their positions and velocities based on their PN accelerations.

It is important to note that the photon’s acceleration is based solely on Newtonian mechanics because the PN expansion assumes that the object’s speed is smaller than the speed of light in vacuum, which is not the case for photons because their speeds are fixed at the speed of light. The photon’s speed is kept constant at the speed of light in vacuum, so the photon’s acceleration only serves the purpose of changing its direction. This is not a general-relativity approach to consider the curvature of spacetime (the photon’s trajectory), but a reasonable estimate based on Newtonian mechanics.

The energy of a photon is obtained based on Planck’s law.

![Equation](./Readme%20equations/eq9.png)

where 𝜈 is the frequency of the light and ℎ is the Planck’s constant. The photon’s “mass” for the purpose of computing acceleration is obtained by Einstein’s mass-energy equivalence equation.

![Equation](./Readme%20equations/eq10.png)

where 𝑐 is the speed of light in vacuum. So the “mass” of a photon is supposedly

![Equation](./Readme%20equations/eq11.png)

The frequency of light used to render the scene is set to 550 THz (the yellow-green region of the visible spectrum).

After obtaining a reasonable image, the program switches to step B to update the positions and velocities of object for the next timestep. It is important to note that the timesteps in step A and step B are independent of each other.

Ray tracing is normally very slow, but in this case when we have an iterative simulation for the photons, it is even worse. In order to have a faster code, we vectorized the operations over the data using Numpy arrays as much as possible. We also used compilation with Numba and jitting, which made the code considerably faster, although it also had some compilation instabilities sometimes. Finally, we implemented the possibility to render only some part of the image, allowing for splitting up the rendering between several processes or computers.

## Results and discussion

We used the following texture for the background (source: ESO/Serge Brunier, Frederic Tapissie):

![Image](./background.jpg)

With only one, static black holes located in the middle of the scene, with a mass of 2e35 kilograms, this is the result we got:

![Image](./out/render%202.png)

We can see very clearly the huge lens effect of the black hole, highly distorting the image of the galaxy behind it. This result is very satisfying because it looks a lot like the ones we can find on the internet.

Here is another one with two black holes this time, with their masses being respectively 2e35 and 4e35 kilograms.

![Image](./out/render%203.png)

We can see that the black holes’ shadows (the black regions, which are actually not the event horizon but contain it) are not round.
We decide a pixel appears as black if it reaches a singularity. Since we are working with numerical approximations, reaching a singularity means passing below a certain distance to it (this distance being computed using the softening parameter). This may seem to be an artificial way to make the shadows appear, as we could think that this naturally defines a black sphere. However, the picture above clearly shows that the rendering of the shadows is more complex than just intersecting a black sphere. The black regions come from the fact that their corresponding photons ended up close to the singularity not necessarily after going to it through a direct straight line, but potentially after orbiting a little bit around it.

A sample video of three black holes with two of them merging is available here https://youtu.be/qUxxHK43KZQ .

The other simulation videos are also made available here:

https://youtu.be/4EVIRs97euc

https://youtu.be/tyOu8qth9vc

https://youtu.be/k7BCGcoyDDM

## Conclusion

Although we used a non-perfectly correct physical approach, using Newtonian methods to simulate the photons and only a first order post-Newtonian approach for the interactions between the black holes, the results we got are surprisingly satisfying, being very close to many simulations we found on the internet, including on the NASA’s website [3].

In order to have an even better-looking result, a possible improvement could be to add an accretion disk around the black holes, which could possibly be simulated using many bright particles orbiting around it.

In conclusion, this project was highly interesting to work on, and allowed us to put into practice the notions we saw in class during this term, in the very practical and fascinating context of black holes.

## References

[1] Post-Newtonian N-body Dynamics, Emiel Por, Minor Master Project, Universiteit Leiden, October 2014. https://home.strw.leidenuniv.nl/~por/docs/Por-2014-Post_Newtonian_N_Body_Dynamics.pdf

[2] A. Verbraeck and E. Eisemann, "Interactive Black-Hole Visualization," in IEEE Transactions on Visualization and Computer Graphics, vol. 27, no. 2, pp. 796-805, Feb. 2021, doi: 10.1109/TVCG.2020.3030452.

[3] NASA, "Computer-Simulated Image of a Supermassive Black Hole", April 2016. https://www.nasa.gov/image-article/computer-simulated-image-of-supermassive-black-hole/
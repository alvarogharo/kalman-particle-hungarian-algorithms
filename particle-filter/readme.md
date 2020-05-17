# Particle filter
The algorithm will create new random particles until the tracking object enters any of them,
in that moment the selection, evaluation, diffusion, etc wil take place.

NOTE: With some random initialization the algorithm could get lost because of the diffusion or
the prediction not updating to the desired position. If this happens, it should generate a new random initialization and find again the
object automatically.

Working example (click image to play):


[![](http://img.youtube.com/vi/8DZfBHvnXRQ/0.jpg)](https://www.youtube.com/watch?v=8DZfBHvnXRQ)

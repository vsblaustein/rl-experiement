# rl-experiment
## Initialization
* Creates a gym environment for the raindrops game.
* Uses a sequential model with an input shape of (100, 60, 4) which comes from the original game frame size 800x480 pixels.
* The neural network has three 2D convolutional layers and uses the rectified linear (ReLU) activation function. 
* Final dense layer has 11 units for the actions the agent can take. 1 unit is the option of doing nothing while the rest allow the agent to move 10, 20, 30, 40, or 50 pixels in either direction. 
## Experiment Defaults
* Epsilon value is initially set to 0.1 and reduced as the training loop iterates to favor exploitation over exploration. 
* Gamma is set to 0.99 for the Q function.
## Training Loop
### Taking Action
* The training loop iterates 1,000,000 times.
* If a random number is less than or equal to the epsilon value or the current frame is less than 3,200, a random action is taken. Otherwise, the model predicts the best action by taking the one with the highest Q value. 
### Observing the Agent after Action
* The action information is obtained with the action chosen and the state.
  * It is a 4 value tuple (state_next, reward, game_over, info).
* If epsilon is greater than 0.0001 and the current frame is greater than 3,200, epsilon is reduced by .0999/300,000. This allows for at least 3,200 frames of selfplay with exploration before reducing the epsilon value to favor exploitation.
### Training the Neural Network
* Training begins once the current frame is greater than 3,200.
* 32 observations are randomly selected. 
* The targets for the Q table are the y values that the neural network predicts. 
  * The targets are obtained from the model with the state list.
* The Q table for the next state is obtained from the model with the state next list.
* Q function is updated for the action of each observation.
  * Q = reward + gamma * (maximum value in the q table for the next state * the inverse of the game_over boolean)
* Loss is calculated with the targets and state list. 
* Loss is updated for the next iteration of the training loop.

This Neural Network is trained using this raindrops game: https://github.com/nnethery/raindrops-gym

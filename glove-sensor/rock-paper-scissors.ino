//
// rock-paper-scissors
//

// logistic regression parameters
float weights[3][3] = {
  {3.007032,-9.370312,6.363242},
  {10.726093,-0.494137,-10.231989},
  {3.132884,-6.927588,3.794688},
};

float biases[3] = {-5.709740,6.722741,-1.013021};

float scaling[3] = {0.000978,0.001600,0.001160};

// setup
void setup() {
  Serial.begin(9600);
  pinMode(A0, INPUT);  
  pinMode(A1, INPUT);  
  pinMode(A2, INPUT);
  pinMode(5, OUTPUT);
}

// the main loop
void loop() {

  // wait for 0.01s
  delay(10);

  // read sensor data
  float d[3];
  d[0] = float(analogRead(A0));
  d[1] = float(analogRead(A1));
  d[2] = float(analogRead(A2));

  // print sensor data
//  Serial.println(String(d[0]) + "," + String(d[1]) + "," + String(d[2])); 
//  return;

  // standardize data
  d[0] *= scaling[0];
  d[1] *= scaling[1];
  d[2] *= scaling[2];

  // calc logits
  float lg[3];
  lg[0] = (d[0] * weights[0][0]) + (d[1] * weights[1][0]) + (d[2] * weights[2][0]) + biases[0];
  lg[1] = (d[0] * weights[0][1]) + (d[1] * weights[1][1]) + (d[2] * weights[2][1]) + biases[1];
  lg[2] = (d[0] * weights[0][2]) + (d[1] * weights[1][2]) + (d[2] * weights[2][2]) + biases[2];

  // control servo
  if (lg[0] > lg[1] && lg[0] > lg[2]) {
    analogWrite(5, 190); // rock -> paper
  } else if (lg[1] > lg[0] && lg[1] > lg[2]) {
    analogWrite(5, 127); // paper -> scissors
  } else {
    analogWrite(5, 50);  // scissors -> rock
  }
}



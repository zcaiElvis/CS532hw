

data {
  vector[N] x;
  vector[N] y;
}


parameters {
  matrix[N, 2] W0; // Since there are only weight and bias
}

model {
  // Priors
  for (i in 1:3){
    w[i] ~ normal(0, 10)
    b[i] ~ normal(0, 10)
  }
}

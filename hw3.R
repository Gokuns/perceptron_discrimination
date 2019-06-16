data_set <- read.csv("hw03_data_set_images.csv", header=FALSE)
label_set <- read.csv("hw03_data_set_labels.csv", header=FALSE)

train_a <- data_set[c(1:25),]
train_b <- data_set[c(40:64),]
train_c <- data_set[c(79:103),]
train_d <- data_set[c(118:142),]
train_e <- data_set[c(157:181),]

test_a <- data_set[c(26:39),]
test_b <- data_set[c(65:78),]
test_c <- data_set[c(104:117),]
test_d <- data_set[c(143:156),]
test_e <- data_set[c(182:195),]

label_a <- as.numeric(label_set[c(1:25),])
label_b <- as.numeric(label_set[c(40:64),])
label_c <- as.numeric(label_set[c(79:103),])
label_d <- as.numeric(label_set[c(118:142),])
label_e <- as.numeric(label_set[c(157:181),])

label_test_a <- as.numeric(label_set[c(26:39),])
label_test_b <- as.numeric(label_set[c(65:78),])
label_test_c <- as.numeric(label_set[c(104:117),])
label_test_d <- as.numeric(label_set[c(143:156),])
label_test_e <- as.numeric(label_set[c(182:195),])

X <- as.matrix(rbind(train_a, train_b, train_c, train_d, train_e))

x_test <- as.matrix(rbind(test_a, test_b, test_c, test_d, test_e))

y_truth <- c(label_a, label_b, label_c, label_d, label_e)

y_truth_test <- c(label_test_a, label_test_b, label_test_c, label_test_d, label_test_e)


label_test_a <- as.numeric(label_set[c(26:39), ])
label_test_b <- as.numeric(label_set[c(65:78), ])
label_test_c <- as.numeric(label_set[c(104:117), ])
label_test_d <- as.numeric(label_set[c(143:156), ])
label_test_e <- as.numeric(label_set[c(182:195), ])

X <- as.matrix(rbind(train_a, train_b, train_c, train_d, train_e))

X_test <- as.matrix(rbind(test_a, test_b, test_c, test_d, test_e))

y_truth <- c(label_a, label_b, label_c, label_d, label_e)

y_truth_test <-
  c(label_test_a,
    label_test_b,
    label_test_c,
    label_test_d,
    label_test_e)


K <- max(y_truth)
N <- length(y_truth)
D <- ncol(X)
Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1
safelog <- function(x) {
  return (log(x + 1e-100))
}



sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

softmax <- function(scores) {
  
  scores <- exp(scores - matrix(apply(scores, MARGIN = 2, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

eta <- 0.005
epsilon <- 1e-3
H <- 20
max_iteration <- 200


set.seed(521)
W <- matrix(runif(H*(D+1), min = -0.01, max = 0.01), D+1, H)
v <- matrix(runif((H + 1) * K, min = -0.01, max = 0.01), H+1, K)

#Z <- sigmoid(cbind(1, X) %*% W)S
Z <- sigmoid(cbind(1, X)%*%W)


#y_predicted <- sigmoid(cbind(1, Z) %*% v)

y_predicted <- softmax(cbind(1, Z) %*% v)


#objective_values <- -sum(y_truth * safelog(y_predicted) + (1 - y_truth) * safelog(1 - y_predicted))
objective_values <- -sum(Y_truth * safelog(y_predicted))

iteration <- 1
while (1) {
    for(k in 1:H+1){
      A <- cbind(1,Z)
      v[k,] <- v[k,] + eta * colSums(((Y_truth - y_predicted) * matrix(A[,k], nrow=nrow(Y_truth), ncol=ncol(v), byrow=FALSE)))
    }
    #v <- v + eta * (y_truth[i] - y_predicted[i]) * c(1, Z[i,])
    for (h in 1:H) {
      #W[k,h] <- W[k,h] + eta * (Y_truth[i] - y_predicted[i]) * v[i,k+1] * Z[, k] * (1 - Z[, k]) * c(1, X[,h])
    W[,h] <- W[,h] + eta * as.matrix(colSums(matrix(rowSums((Y_truth - y_predicted) * matrix(v[h+1,], nrow=nrow(Y_truth), ncol(v), byrow=TRUE)) * as.matrix(Z[,h]) * as.matrix((1 - Z[,h])), nrow=nrow(Y_truth), ncol=nrow(W), byrow=FALSE) * cbind(1,X)))
    }
  
  
  Z <- sigmoid(cbind(1, X) %*% W)
  y_predicted <- softmax(cbind(1, Z) %*% v)
  objective_values <- c(objective_values, -sum(Y_truth * safelog(y_predicted)))
  
  if (objective_values[iteration] < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}
#print(W)

plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

Y_predicted <- apply(y_predicted, 1, which.max)
confusion_matrix <- table(Y_predicted, y_truth)

Z_test <- sigmoid(cbind(1, X_test)%*%W)
y_predicted_test <- softmax(cbind(1, Z_test) %*% v)

Y_predicted_test <- apply(y_predicted_test, 1, which.max)
confusion_matrix1 <- table(Y_predicted_test, y_truth_test)
print(confusion_matrix)
print(confusion_matrix1)




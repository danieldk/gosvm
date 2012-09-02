// Package gosvm trains and applies Support Vector Machines (SVMs)
//
// The package is a binding against libsvm with a Go-ish interface. It
// supports C-SVC, nu-SVC, epsilon-SVR, nu-SVR, and one-class SVMs with
// linear, polynomial, RBF, and sigmoid kernels. Trained models can be
// saved to and loaded from disk, to avoid the (potentially) costly training
// process.
//
// A model is trained using a problem. A problem consists of training
// instances, where each training instance has a class label and a feature
// vector. The training procedure attempts to find one or more functions
// that separate the instances of two classes. This model can then predict
// the class of unseen instances.
//
// Consider for instance that we would like to do sentiment analysis,
// using the following, humble, training corpus:
//
//     Positive: A beautiful album.
//     Negative: A crappy ugly album.
//
// To represent this as an SVM problem, we have to convert the classses
// (positive/negative) to an integral class labels and extract features.
// In this case, we can simply label the classes as positive: 0,
// negative: 1. We will use the words as our features (a: 1, beautiful:
// 2, album: 3, crappy: 4, ugly: 5) and use booleans as our feature values.
// In other words, the sentences will have the following feature vectors:
//
//                 1   2   3   4   5
//               +---+---+---+---+---+
//     Positive: | 1 | 1 | 1 | 0 | 0 |
//               +---+---+---+---+---+
//
//               +---+---+---+---+---+
//     Negative: | 1 | 0 | 1 | 1 | 1 |
//               +---+---+---+---+---+
//
// We can now construct the problem using this representation:
//
//     problem := gosvm.NewProblem()
//     problem.Add(gosvm.TrainingInstance{0, gosvm.FromDenseVector([]float64{1, 1, 1, 0, 0})})
//     problem.Add(gosvm.TrainingInstance{1, gosvm.FromDenseVector([]float64{1, 0, 1, 1, 1})})
//
// The problem is used to train the SVM using a set of parameters to choose
// the SVM type, the kernel, etc. We will use the default parameters, which
// train a C-SVC using a linear kernel:
//
//     param := gosvm.DefaultParameters()
//     model, err := gosvm.TrainModel(param, problem)
//     if err != nil {
//     	log.Fatal(err)
//     }
//
// Of course, now we would like to use this model to classify other
// sentences. For instance:
//
//     This is a beautiful book.
//
// We map this sentence to the feature vector that we used during
// training, simply ignoring words that we did not encounter while training
// the model:
//
//               +---+---+---+---+---+
//     ????????: | 1 | 1 | 0 | 0 | 0 |
//               +---+---+---+---+---+
//
// The Predict method of the model is used to predict the label of this
// feature vector.
//
//     label := model.Predict(gosvm.FromDenseVector([]float64{1, 1, 0, 0, 0}))
//
// As expected, the model will predict the sentence to be positive (0).
package gosvm

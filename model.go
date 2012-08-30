package libsvm

/*
#cgo LDFLAGS: -lsvm
#include "wrap.h"
*/
import "C"

import (
	"runtime"
)

type Model struct {
	model *C.svm_model_t
	// Keep a pointer to the problem, since C model depends on it.
	problem *Problem
}

func TrainModel(problem *Problem) *Model {
	param := C.parameter_new()
	cmodel := C.svm_train_wrap(problem.problem, param)
	model := &Model{cmodel, problem}
	runtime.SetFinalizer(model, finalizeModel)
	return model
}

func (model *Model) Save() {
	C.svm_save_model_wrap(model.model)
}

func finalizeModel(model *Model) {
	println("finalizing the model")
	C.svm_free_and_destroy_model_wrap(model.model)
	model.problem = nil
}

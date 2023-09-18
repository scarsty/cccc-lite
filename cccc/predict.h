#pragma once

#include "dll_export.h"

EXTERN_C_BEGIN

typedef void* Predictor;

DLL_EXPORT int predict_data(const Predictor predictor, void* x, void* a, int n);
DLL_EXPORT Predictor create_predictor();
DLL_EXPORT int destroy_predictor(Predictor predictor);
DLL_EXPORT int init_predictor(const Predictor predictor, const char* ini, int size);
DLL_EXPORT int get_predictor_size(const Predictor predictor, const int** dim_x, int* size_dim_x, int* size_x, const int** dim_y, int* size_dim_y, int* size_y);
DLL_EXPORT int load_predictor_weight(const Predictor predictor, void* weight, int size);

EXTERN_C_END

/*
    This interface is used to test data with trained neural net.
    You have to prepare the data and malloc the space to save the results yourself.

    An example should like this:

    PredictorPointer predictor;
    create_predictor(&predictor);
    init_predictor(predictor, ini_file);
    load_predictor_weight(predictor, weight, size);

    float input[......], output[......];
    fill data in input......

    predict_data(predictor, input, ......, output, ......);

    result will be save in output, do something with it.

    destory_predictor(predictor);

*/

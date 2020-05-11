/* woco.i */
%module woco

%include "std_string.i"
%include "std_vector.i"

#define DLL_EXPORT

%{
#define SWIG_FILE_WITH_INIT
#include "../woco/types.h"
#include "../woco/Option.h"
#include "../woco/CudaControl.h"
#include "../woco/Matrix.h"
#include "../woco/MatrixExtend.h"
#include "../woco/MatrixOperator.h"
#include "../woco/Net.h"
#include "../woco/DataPreparer.h"
#include "../woco/Neural.h"
using namespace woco;
%}


%include "../woco/types.h"
%include "../woco/Option.h"
%include "../woco/CudaControl.h"
%include "../woco/Matrix.h"
%include "../woco/MatrixExtend.h"
%include "../woco/MatrixOperator.h"
%include "../woco/Net.h"
%include "../woco/DataPreparer.h"
%include "../woco/Neural.h"
using namespace woco;

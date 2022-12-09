

#MOTOR CONSTANTS
TRAIN_MSE_REAL  = 0
TRAIN_MSE_IMAG  = 1
TEST_MSE        = 2
TRAIN_EQ        = 3
TEST_EQ         = 4
TRAIN_COMPLETE  = 5
TEST_COMPLETE   = 6
TRAIN_MSE_MAG   = 7 
TRAIN_MSE_PHASE = 8 
TRAIN_MSE_INV   = 9
TEST_POLAR      = 10
TEST_MSE_INV    = 11
TRAIN_BCE       = 12
TEST_BCE        = 13

#real_imag

REAL  = 0
IMAG  = 1
BOTH  = 2
ABS   = 3
ANGLE = 4
INV   = 5
FOUR  = 6

real_imag_str = {
    REAL : "REAL", 
    IMAG : "IMAG", 
    BOTH : "BOTH", 
    ABS  : "ABS",  
    ANGLE : "ANGLE",
    INV   : "LMSE",
    FOUR  : "FOUR",
}

#loss_type

MSE = 0
CROSSENTROPY = 1
MSE_COMPLETE = 2
MSE_INV = 3
BCE = 4
CUSTOM = 5

#GET Y 
GET_TOGGLE = 0
GET_PLANE  = 1
GET_LMSE   = 2
GET_ANGLE  = 3
GET_MAGNITUD = 4


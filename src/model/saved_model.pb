͟
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02unknown8Ը
~
conv2d/kernelVarHandleOp*
dtype0*
shared_nameconv2d/kernel*
shape: *
_output_shapes
: 
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
shape: *
dtype0*
shared_nameconv2d/bias*
_output_shapes
: 
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
: 
?
conv2d_1/kernelVarHandleOp* 
shared_nameconv2d_1/kernel*
_output_shapes
: *
shape: @*
dtype0
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
: @
r
conv2d_1/biasVarHandleOp*
dtype0*
shape:@*
shared_nameconv2d_1/bias*
_output_shapes
: 
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
u
dense/kernelVarHandleOp*
shared_namedense/kernel*
dtype0*
_output_shapes
: *
shape:	?

n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
shape:
*
shared_name
dense/bias*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
shape: *
shared_name	Adam/iter*
dtype0	
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
shared_nameAdam/beta_1*
shape: *
_output_shapes
: *
dtype0
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
shared_nameAdam/beta_2*
dtype0*
_output_shapes
: *
shape: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
dtype0*
shape: *
_output_shapes
: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
dtype0*
shape: *#
shared_nameAdam/learning_rate*
_output_shapes
: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
shared_nametotal*
_output_shapes
: *
dtype0*
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
?
Adam/conv2d/kernel/mVarHandleOp*
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/m*
_output_shapes
: 
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m*
_output_shapes
: 
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
shape: @*
_output_shapes
: *
dtype0*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*$
shared_nameAdam/dense/kernel/m*
shape:	?

|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
dtype0*"
shared_nameAdam/dense/bias/m*
_output_shapes
: *
shape:

s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
dtype0*
_output_shapes
:

?
Adam/conv2d/kernel/vVarHandleOp*
shape: *
dtype0*
_output_shapes
: *%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v*
_output_shapes
: 
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*'
shared_nameAdam/conv2d_1/kernel/v*
shape: @*
_output_shapes
: *
dtype0
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*
dtype0*&
_output_shapes
: @
?
Adam/conv2d_1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *%
shared_nameAdam/conv2d_1/bias/v*
shape:@
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
shape:	?
*
_output_shapes
: *$
shared_nameAdam/dense/kernel/v*
dtype0
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
dtype0*
_output_shapes
:	?

z
Adam/dense/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:
*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?.
ConstConst"/device:CPU:0*?.
value?.B?. B?-
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
R
#regularization_losses
$	variables
%trainable_variables
&	keras_api
R
'regularization_losses
(	variables
)trainable_variables
*	keras_api
R
+regularization_losses
,	variables
-trainable_variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
?
5iter

6beta_1

7beta_2
	8decay
9learning_ratemjmkmlmm/mn0movpvqvrvs/vt0vu
 
*
0
1
2
3
/4
05
*
0
1
2
3
/4
05
?

regularization_losses
	variables

:layers
;metrics
<non_trainable_variables
=layer_regularization_losses
trainable_variables
 
 
 
 
?
regularization_losses
	variables

>layers
?metrics
@non_trainable_variables
Alayer_regularization_losses
trainable_variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
	variables

Blayers
Cmetrics
Dnon_trainable_variables
Elayer_regularization_losses
trainable_variables
 
 
 
?
regularization_losses
	variables

Flayers
Gmetrics
Hnon_trainable_variables
Ilayer_regularization_losses
trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
 	variables

Jlayers
Kmetrics
Lnon_trainable_variables
Mlayer_regularization_losses
!trainable_variables
 
 
 
?
#regularization_losses
$	variables

Nlayers
Ometrics
Pnon_trainable_variables
Qlayer_regularization_losses
%trainable_variables
 
 
 
?
'regularization_losses
(	variables

Rlayers
Smetrics
Tnon_trainable_variables
Ulayer_regularization_losses
)trainable_variables
 
 
 
?
+regularization_losses
,	variables

Vlayers
Wmetrics
Xnon_trainable_variables
Ylayer_regularization_losses
-trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
?
1regularization_losses
2	variables

Zlayers
[metrics
\non_trainable_variables
]layer_regularization_losses
3trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6

^0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	_total
	`count
a
_fn_kwargs
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

_0
`1
 
?
bregularization_losses
c	variables

flayers
gmetrics
hnon_trainable_variables
ilayer_regularization_losses
dtrainable_variables
 
 

_0
`1
 
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
?
serving_default_input_1Placeholder*$
shape:?????????*/
_output_shapes
:?????????*
dtype0
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/bias*'
_output_shapes
:?????????
*
Tout
2*
Tin
	2*,
_gradient_op_typePartitionedCall-23912*,
f'R%
#__inference_signature_wrapper_23695**
config_proto

CPU

GPU 2J 8
O
saver_filenamePlaceholder*
dtype0*
shape: *
_output_shapes
: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*
_output_shapes
: *
Tout
2*,
_gradient_op_typePartitionedCall-23959*&
Tin
2	**
config_proto

CPU

GPU 2J 8*'
f"R 
__inference__traced_save_23958
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/v*
_output_shapes
: *
Tout
2**
config_proto

CPU

GPU 2J 8*%
Tin
2*,
_gradient_op_typePartitionedCall-24047**
f%R#
!__inference__traced_restore_24046??
?
I
-__inference_max_pooling2d_layer_call_fn_23450

inputs
identity?
PartitionedCallPartitionedCallinputs*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23441**
config_proto

CPU

GPU 2J 8*J
_output_shapes8
6:4????????????????????????????????????*
Tout
2*,
_gradient_op_typePartitionedCall-23447*
Tin
2?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_23552

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:??????????*
T0\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23483

inputs
identity?
MaxPoolMaxPoolinputs*
paddingVALID*
ksize
*
strides
*J
_output_shapes8
6:4????????????????????????????????????{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4????????????????????????????????????*
T0"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_23637

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*/
_output_shapes
:????????? *
Tout
2*,
_gradient_op_typePartitionedCall-23428**
config_proto

CPU

GPU 2J 8*
Tin
2*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_23422?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23441**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:????????? *,
_gradient_op_typePartitionedCall-23447*
Tout
2*
Tin
2?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23464*,
_gradient_op_typePartitionedCall-23470**
config_proto

CPU

GPU 2J 8*
Tout
2*/
_output_shapes
:?????????@*
Tin
2?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@*
Tin
2**
config_proto

CPU

GPU 2J 8*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23483*
Tout
2*,
_gradient_op_typePartitionedCall-23489?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-23518*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23512*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2*(
_output_shapes
:???????????
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-23556*
Tout
2*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23545*
Tin
2?
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23580*
Tout
2*'
_output_shapes
:?????????
*
Tin
2**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-23586?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: 
?
?
&__inference_conv2d_layer_call_fn_23433

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23428*A
_output_shapes/
-:+??????????????????????????? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_23422**
config_proto

CPU

GPU 2J 8*
Tout
2*
Tin
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+??????????????????????????? *
T0"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_23830

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:??????????*
T0\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_23835

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*(
_output_shapes
:??????????*,
_gradient_op_typePartitionedCall-23556*
Tout
2*
Tin
2**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23545?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_23475

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+???????????????????????????@*,
_gradient_op_typePartitionedCall-23470*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23464?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_23851

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0V
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?a
?
!__inference__traced_restore_24046
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count,
(assignvariableop_13_adam_conv2d_kernel_m*
&assignvariableop_14_adam_conv2d_bias_m.
*assignvariableop_15_adam_conv2d_1_kernel_m,
(assignvariableop_16_adam_conv2d_1_bias_m+
'assignvariableop_17_adam_dense_kernel_m)
%assignvariableop_18_adam_dense_bias_m,
(assignvariableop_19_adam_conv2d_kernel_v*
&assignvariableop_20_adam_conv2d_bias_v.
*assignvariableop_21_adam_conv2d_1_kernel_v,
(assignvariableop_22_adam_conv2d_1_bias_v+
'assignvariableop_23_adam_dense_kernel_v)
%assignvariableop_24_adam_dense_bias_v
identity_26??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE?
RestoreV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0z
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0~
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:}
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0	|
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
dtype0	*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:~
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:~
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0}
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0{
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:{
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_conv2d_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_conv2d_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype0P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_conv2d_1_kernel_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
_output_shapes
:*
T0?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_conv2d_1_bias_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_kernel_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_dense_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_kernel_vIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv2d_bias_vIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_1_kernel_vIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_1_bias_vIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_kernel_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp%assignvariableop_24_adam_dense_bias_vIdentity_24:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0?
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : 
?	
?
*__inference_sequential_layer_call_fn_23783

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_23637*,
_gradient_op_typePartitionedCall-23638*
Tout
2*
Tin
	2*'
_output_shapes
:?????????
**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_23545

inputs
identity?Q
dropout/rateConst*
dtype0*
valueB
 *   ?*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: _
dropout/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*
T0*(
_output_shapes
:???????????
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:???????????
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*(
_output_shapes
:??????????*
T0R
dropout/sub/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:??????????*
T0b
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:??????????*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*(
_output_shapes
:??????????*

DstT0j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*(
_output_shapes
:??????????*
T0Z
IdentityIdentitydropout/mul_1:z:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_23647
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*
Tout
2*,
_gradient_op_typePartitionedCall-23638*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_23637*
Tin
	2*'
_output_shapes
:?????????
**
config_proto

CPU

GPU 2J 8?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : 
?"
?
E__inference_sequential_layer_call_and_return_conditional_losses_23772

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*
paddingVALID*/
_output_shapes
:????????? *
strides
?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*
strides
*
paddingVALID*
ksize
*/
_output_shapes
:????????? ?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
strides
*
paddingVALID?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
paddingVALID*/
_output_shapes
:?????????@*
ksize
*
strides
f
flatten/Reshape/shapeConst*
dtype0*
valueB"????@  *
_output_shapes
:?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0i
dropout/IdentityIdentityflatten/Reshape:output:0*(
_output_shapes
:??????????*
T0?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0?
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentitydense/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : 
?	
?
*__inference_sequential_layer_call_fn_23678
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:?????????
*,
_gradient_op_typePartitionedCall-23669*
Tout
2*
Tin
	2*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_23668?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_1: 
?
K
/__inference_max_pooling2d_1_layer_call_fn_23492

inputs
identity?
PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-23489*J
_output_shapes8
6:4????????????????????????????????????*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23483**
config_proto

CPU

GPU 2J 8*
Tout
2*
Tin
2?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_23800

inputs
identity^
Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????@  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_23840

inputs
identity?
PartitionedCallPartitionedCallinputs*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23552*
Tin
2*,
_gradient_op_typePartitionedCall-23564*(
_output_shapes
:??????????**
config_proto

CPU

GPU 2J 8*
Tout
2a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_23805

inputs
identity?
PartitionedCallPartitionedCallinputs*(
_output_shapes
:??????????*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23512*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*,
_gradient_op_typePartitionedCall-23518a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_23617
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23428*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_23422*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:????????? ?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tout
2*/
_output_shapes
:????????? *
Tin
2*,
_gradient_op_typePartitionedCall-23447*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23441?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23464*/
_output_shapes
:?????????@*,
_gradient_op_typePartitionedCall-23470*
Tin
2*
Tout
2?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-23489*
Tout
2*/
_output_shapes
:?????????@*
Tin
2*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23483?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*(
_output_shapes
:??????????*,
_gradient_op_typePartitionedCall-23518*
Tout
2*
Tin
2*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23512**
config_proto

CPU

GPU 2J 8?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*,
_gradient_op_typePartitionedCall-23564*
Tin
2**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23552*
Tout
2*(
_output_shapes
:???????????
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23580*
Tout
2*,
_gradient_op_typePartitionedCall-23586*'
_output_shapes
:?????????
*
Tin
2?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall: :' #
!
_user_specified_name	input_1: : : : : 
?6
?

__inference__traced_save_23958
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*<
value3B1 B+_temp_bf749f000468449baba93877e09c8713/part*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0?
SaveV2/shape_and_slicesConst"/device:CPU:0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:*
dtype0?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop"/device:CPU:0*'
dtypes
2	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHq
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:	?
:
: : : : : : : : : : @:@:	?
:
: : : @:@:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : 
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_23512

inputs
identity^
Reshape/shapeConst*
valueB"????@  *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:??????????*
T0Y
IdentityIdentityReshape:output:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*.
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_23858

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-23586**
config_proto

CPU

GPU 2J 8*
Tin
2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23580*'
_output_shapes
:?????????
*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
?)
?
 __inference__wrapped_model_23408
input_14
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: ?
sequential/conv2d/Conv2DConv2Dinput_1/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: ?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? *
T0|
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*/
_output_shapes
:????????? *
strides
*
paddingVALID*
ksize
?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0?
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:?????????@*
paddingVALID*
strides
*
T0?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????@*
T0?
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
"sequential/max_pooling2d_1/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*
strides
*
paddingVALID*/
_output_shapes
:?????????@*
ksize
q
 sequential/flatten/Reshape/shapeConst*
valueB"????@  *
dtype0*
_output_shapes
:?
sequential/flatten/ReshapeReshape+sequential/max_pooling2d_1/MaxPool:output:0)sequential/flatten/Reshape/shape:output:0*
T0*(
_output_shapes
:??????????
sequential/dropout/IdentityIdentity#sequential/flatten/Reshape:output:0*
T0*(
_output_shapes
:???????????
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0?
sequential/dense/MatMulMatMul$sequential/dropout/Identity:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0x
sequential/dense/SoftmaxSoftmax!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentity"sequential/dense/Softmax:softmax:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp:' #
!
_user_specified_name	input_1: : : : : : 
?
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23441

inputs
identity?
MaxPoolMaxPoolinputs*
ksize
*J
_output_shapes8
6:4????????????????????????????????????*
strides
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:& "
 
_user_specified_nameinputs
?
?
A__inference_conv2d_layer_call_and_return_conditional_losses_23422

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingVALID*
T0*
strides
*A
_output_shapes/
-:+??????????????????????????? ?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? ?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? "
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_23825

inputs
identity?Q
dropout/rateConst*
_output_shapes
: *
valueB
 *   ?*
dtype0C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ??*
dtype0?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*
T0*(
_output_shapes
:???????????
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:???????????
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:??????????R
dropout/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:??????????b
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:??????????*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*(
_output_shapes
:??????????*

DstT0j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*(
_output_shapes
:??????????*
T0Z
IdentityIdentitydropout/mul_1:z:0*(
_output_shapes
:??????????*
T0"
identityIdentity:output:0*'
_input_shapes
:??????????:& "
 
_user_specified_nameinputs
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_23580

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	?
i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:?????????
*
T0V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_23598
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?dropout/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_23422*,
_gradient_op_typePartitionedCall-23428*
Tin
2*/
_output_shapes
:????????? **
config_proto

CPU

GPU 2J 8*
Tout
2?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-23447*/
_output_shapes
:????????? *
Tout
2*
Tin
2*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23441?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23464*,
_gradient_op_typePartitionedCall-23470**
config_proto

CPU

GPU 2J 8*
Tout
2*/
_output_shapes
:?????????@?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23483*
Tout
2*/
_output_shapes
:?????????@**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-23489?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23512*(
_output_shapes
:??????????*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-23518?
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:??????????*
Tin
2*,
_gradient_op_typePartitionedCall-23556*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23545?
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tout
2*'
_output_shapes
:?????????
*
Tin
2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23580*,
_gradient_op_typePartitionedCall-23586?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:' #
!
_user_specified_name	input_1: : : : : : 
?
?
#__inference_signature_wrapper_23695
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*,
_gradient_op_typePartitionedCall-23686*'
_output_shapes
:?????????
**
config_proto

CPU

GPU 2J 8*
Tin
	2*)
f$R"
 __inference__wrapped_model_23408*
Tout
2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :' #
!
_user_specified_name	input_1: : : 
?2
?
E__inference_sequential_layer_call_and_return_conditional_losses_23742

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: ?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*
T0*/
_output_shapes
:????????? ?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: ?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:????????? *
T0f
conv2d/ReluReluconv2d/BiasAdd:output:0*/
_output_shapes
:????????? *
T0?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:????????? *
strides
*
ksize
*
paddingVALID?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*
paddingVALID*/
_output_shapes
:?????????@*
strides
?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:?????????@*
T0j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
strides
*
paddingVALID*
ksize
*/
_output_shapes
:?????????@f
flatten/Reshape/shapeConst*
_output_shapes
:*
valueB"????@  *
dtype0?
flatten/ReshapeReshape max_pooling2d_1/MaxPool:output:0flatten/Reshape/shape:output:0*(
_output_shapes
:??????????*
T0Y
dropout/dropout/rateConst*
_output_shapes
: *
valueB
 *   ?*
dtype0]
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: g
"dropout/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ???
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
dtype0*(
_output_shapes
:??????????*
T0?
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0?
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*(
_output_shapes
:??????????*
T0?
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*(
_output_shapes
:??????????*
T0Z
dropout/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
_output_shapes
: *
T0^
dropout/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*(
_output_shapes
:???????????
dropout/dropout/mulMulflatten/Reshape:output:0dropout/dropout/truediv:z:0*(
_output_shapes
:??????????*
T0?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

SrcT0
*(
_output_shapes
:??????????*

DstT0?
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	?
*
dtype0?
dense/MatMulMatMuldropout/dropout/mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*'
_output_shapes
:?????????
*
T0?
IdentityIdentitydense/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_23668

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_23422**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:????????? *
Tin
2*,
_gradient_op_typePartitionedCall-23428?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23441*
Tin
2*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:????????? *,
_gradient_op_typePartitionedCall-23447?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tout
2*/
_output_shapes
:?????????@*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23464*,
_gradient_op_typePartitionedCall-23470*
Tin
2?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23483*
Tout
2*
Tin
2*,
_gradient_op_typePartitionedCall-23489**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:?????????@?
flatten/PartitionedCallPartitionedCall(max_pooling2d_1/PartitionedCall:output:0*
Tin
2*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_23512*,
_gradient_op_typePartitionedCall-23518**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:??????????*
Tout
2?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_23552**
config_proto

CPU

GPU 2J 8*
Tout
2*(
_output_shapes
:??????????*,
_gradient_op_typePartitionedCall-23564?
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2**
config_proto

CPU

GPU 2J 8*
Tout
2*,
_gradient_op_typePartitionedCall-23586*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_23580*'
_output_shapes
:?????????
?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall*'
_output_shapes
:?????????
*
T0"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : : : 
?	
?
*__inference_sequential_layer_call_fn_23794

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6**
config_proto

CPU

GPU 2J 8*,
_gradient_op_typePartitionedCall-23669*
Tout
2*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_23668*'
_output_shapes
:?????????
*
Tin
	2?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : 
?
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23464

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingVALID*
T0*A
_output_shapes/
-:+???????????????????????????@*
strides
?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+???????????????????????????@*
T0j
ReluReluBiasAdd:output:0*A
_output_shapes/
-:+???????????????????????????@*
T0?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*A
_output_shapes/
-:+???????????????????????????@*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????9
dense0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?,
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
v__call__
*w&call_and_return_all_conditional_losses
x_default_save_signature"?)
_tf_keras_sequential?){"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 28, 28, 1]}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "batch_input_shape": [null, 28, 28, 1]}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
y__call__
*z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "input_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28, 28, 1], "config": {"batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "sparse": false, "name": "input_1"}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
{__call__
*|&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
?
#regularization_losses
$	variables
%trainable_variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
'regularization_losses
(	variables
)trainable_variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
+regularization_losses
,	variables
-trainable_variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1600}}}}
?
5iter

6beta_1

7beta_2
	8decay
9learning_ratemjmkmlmm/mn0movpvqvrvs/vt0vu"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
/4
05"
trackable_list_wrapper
J
0
1
2
3
/4
05"
trackable_list_wrapper
?

regularization_losses
	variables

:layers
;metrics
<non_trainable_variables
=layer_regularization_losses
trainable_variables
v__call__
x_default_save_signature
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
	variables

>layers
?metrics
@non_trainable_variables
Alayer_regularization_losses
trainable_variables
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
':% 2conv2d/kernel
: 2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
	variables

Blayers
Cmetrics
Dnon_trainable_variables
Elayer_regularization_losses
trainable_variables
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
	variables

Flayers
Gmetrics
Hnon_trainable_variables
Ilayer_regularization_losses
trainable_variables
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_1/kernel
:@2conv2d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
 	variables

Jlayers
Kmetrics
Lnon_trainable_variables
Mlayer_regularization_losses
!trainable_variables
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
#regularization_losses
$	variables

Nlayers
Ometrics
Pnon_trainable_variables
Qlayer_regularization_losses
%trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
'regularization_losses
(	variables

Rlayers
Smetrics
Tnon_trainable_variables
Ulayer_regularization_losses
)trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
+regularization_losses
,	variables

Vlayers
Wmetrics
Xnon_trainable_variables
Ylayer_regularization_losses
-trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?
2dense/kernel
:
2
dense/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
1regularization_losses
2	variables

Zlayers
[metrics
\non_trainable_variables
]layer_regularization_losses
3trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
'
^0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	_total
	`count
a
_fn_kwargs
bregularization_losses
c	variables
dtrainable_variables
e	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
bregularization_losses
c	variables

flayers
gmetrics
hnon_trainable_variables
ilayer_regularization_losses
dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:, @2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
$:"	?
2Adam/dense/kernel/m
:
2Adam/dense/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:, @2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
$:"	?
2Adam/dense/kernel/v
:
2Adam/dense/bias/v
?2?
*__inference_sequential_layer_call_fn_23794
*__inference_sequential_layer_call_fn_23783
*__inference_sequential_layer_call_fn_23678
*__inference_sequential_layer_call_fn_23647?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_23598
E__inference_sequential_layer_call_and_return_conditional_losses_23617
E__inference_sequential_layer_call_and_return_conditional_losses_23742
E__inference_sequential_layer_call_and_return_conditional_losses_23772?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_23408?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
&__inference_conv2d_layer_call_fn_23433?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
A__inference_conv2d_layer_call_and_return_conditional_losses_23422?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
-__inference_max_pooling2d_layer_call_fn_23450?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23441?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_conv2d_1_layer_call_fn_23475?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23464?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
/__inference_max_pooling2d_1_layer_call_fn_23492?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23483?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
'__inference_flatten_layer_call_fn_23805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_23800?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_23840
'__inference_dropout_layer_call_fn_23835?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_23825
B__inference_dropout_layer_call_and_return_conditional_losses_23830?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_23858?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_23851?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
2B0
#__inference_signature_wrapper_23695input_1
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
#__inference_signature_wrapper_23695|/0C?@
? 
9?6
4
input_1)?&
input_1?????????"-?*
(
dense?
dense?????????
?
B__inference_dropout_layer_call_and_return_conditional_losses_23825^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? |
'__inference_dropout_layer_call_fn_23840Q4?1
*?'
!?
inputs??????????
p 
? "????????????
-__inference_max_pooling2d_layer_call_fn_23450?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_conv2d_1_layer_call_and_return_conditional_losses_23464?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
*__inference_sequential_layer_call_fn_23678d/0@?=
6?3
)?&
input_1?????????
p 

 
? "??????????
?
@__inference_dense_layer_call_and_return_conditional_losses_23851]/00?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? ?
 __inference__wrapped_model_23408q/08?5
.?+
)?&
input_1?????????
? "-?*
(
dense?
dense?????????
?
(__inference_conv2d_1_layer_call_fn_23475?I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
/__inference_max_pooling2d_1_layer_call_fn_23492?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
*__inference_sequential_layer_call_fn_23647d/0@?=
6?3
)?&
input_1?????????
p

 
? "??????????
?
E__inference_sequential_layer_call_and_return_conditional_losses_23598q/0@?=
6?3
)?&
input_1?????????
p

 
? "%?"
?
0?????????

? ?
*__inference_sequential_layer_call_fn_23783c/0??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
&__inference_conv2d_layer_call_fn_23433?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? 
'__inference_flatten_layer_call_fn_23805T7?4
-?*
(?%
inputs?????????@
? "????????????
A__inference_conv2d_layer_call_and_return_conditional_losses_23422?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
B__inference_flatten_layer_call_and_return_conditional_losses_23800a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_23772p/0??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
B__inference_dropout_layer_call_and_return_conditional_losses_23830^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? |
'__inference_dropout_layer_call_fn_23835Q4?1
*?'
!?
inputs??????????
p
? "????????????
*__inference_sequential_layer_call_fn_23794c/0??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_23441?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_23742p/0??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
E__inference_sequential_layer_call_and_return_conditional_losses_23617q/0@?=
6?3
)?&
input_1?????????
p 

 
? "%?"
?
0?????????

? ?
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_23483?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? y
%__inference_dense_layer_call_fn_23858P/00?-
&?#
!?
inputs??????????
? "??????????

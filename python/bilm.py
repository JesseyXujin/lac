#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import paddle.fluid.layers as layers
import paddle.fluid as fluid
import numpy as np
import ipdb
test_mode=False
random_seed=123
para_init=False
cell_clip=3.0
proj_clip=3.0
init1=0.1
hidden_size=4096
para_init=False
vocab_size=52445
emb_size=512
modify =1

def dropout(input):
    dropout1=0.1
    if modify == 1:
       dropout1=0.5
    return layers.dropout(
            input,
            dropout_prob=dropout1,
            dropout_implementation="upscale_in_train",
            seed=random_seed,
            is_test=False)


def lstmp_encoder(input_seq, gate_size, h_0, c_0, para_name, proj_size, test_mode, args):
    # A lstm encoder implementation with projection.
    # Linear transformation part for input gate, output gate, forget gate
    # and cell activation vectors need be done outside of dynamic_lstm.
    # So the output size is 4 times of gate_size.

    if para_init:
        init = fluid.initializer.Constant(init1)
        init_b = fluid.initializer.Constant(0.0)
    else:
        init = None
        init_b = None
    if modify==0:
        input_seq = dropout(input_seq)
    input_proj = layers.fc(input=input_seq,
                           param_attr=fluid.ParamAttr(
                               name=para_name + '_gate_w', initializer=init),
                           size=gate_size * 4,
                           act=None,
                           bias_attr=False)
    #layers.Print(input_seq, message='input_seq', summarize=10)
    #layers.Print(input_proj, message='input_proj', summarize=10)
    hidden, cell = layers.dynamic_lstmp(
        input=input_proj,
        size=gate_size * 4,
        proj_size=proj_size,
        h_0=h_0,
        c_0=c_0,
        use_peepholes=False,
        proj_clip=proj_clip,
        cell_clip=cell_clip,
        proj_activation="identity",
        param_attr=fluid.ParamAttr(initializer=init),
        bias_attr=fluid.ParamAttr(initializer=init_b))

    return hidden, cell, input_proj
def emb(x, vocab_size=52445,emb_size=512):
    x_emb = layers.embedding(
        input=x,
        size=[vocab_size, emb_size],
        dtype='float32',
        is_sparse=False,
        param_attr=fluid.ParamAttr(name='embedding_para'))
    
    return x_emb


def encoder_1(x_emb,
            vocab_size,
            emb_size,
            init_hidden=None,
            init_cell=None,
            para_name='',
            args=None):
    rnn_input = x_emb
    #rnn_input.stop_gradient = True
    rnn_outs = []
    rnn_outs_ori = []
    cells = []
    projs = []
    num_layers=2
    for i in range(num_layers):
        if modify==0:
            rnn_input = dropout(rnn_input)
        if init_hidden and init_cell:
            h0 = layers.squeeze(
                layers.slice(
                    init_hidden, axes=[0], starts=[i], ends=[i + 1]),
                axes=[0])
            c0 = layers.squeeze(
                layers.slice(
                    init_cell, axes=[0], starts=[i], ends=[i + 1]),
                axes=[0])
        else:
            h0 = c0 = None
        rnn_out, cell, input_proj = lstmp_encoder(
            rnn_input, hidden_size, h0, c0,
            para_name + 'layer{}'.format(i + 1), emb_size, test_mode, args)
        rnn_out_ori = rnn_out
        if i > 0:
            rnn_out = rnn_out + rnn_input
        if modify==0:
            rnn_out = dropout(rnn_out)
        rnn_out.stop_gradient = True
        rnn_outs.append(rnn_out)
        #rnn_outs_ori.stop_gradient = True
        rnn_outs_ori.append(rnn_out_ori)
    #ipdb.set_trace()
     #layers.Print(input_seq, message='input_seq', summarize=10)
    #print(len(rnn_outs))
    a1 = layers.create_parameter(
        [1], dtype="float32", name="gamma1")
    a2 = layers.create_parameter(
        [1], dtype="float32", name="gamma2")
    rnn_outs[0].stop_gradient=True
    rnn_outs[1].stop_gradient=True
    num_layer1=rnn_outs[0]*a1
    num_layer2=rnn_outs[1]*a2
    output_layer=num_layer1*0.5+num_layer2*0.5
    #layers.Print(rnn_outs[1], message='rnn_outs1', summarize=10)
    #layers.Print(rnn_outs[0], message='rnn_outs0', summarize=10)
    #layers.Print(num_layer1, message='num_layer1', summarize=10)
    #layers.Print(num_layer2, message='num_layer2', summarize=10)
    #layers.Print(output_layer, message='output_layer', summarize=10)
    #layers.Print(a1, message='a1', summarize=10)
    #layers.Print(a2, message='a2', summarize=10)
    return  output_layer, rnn_outs_ori


def elmo_encoder(x_emb,x_emb_r):
    #args modify
    emb_size = 512
    proj_size = 512
    hidden_size = 4096
    #batch_size = 32
    num_layers = 2
    num_steps = 20

    lstm_outputs = []
    #layers.Print(x_emb, message='x_emb', summarize=10)
    #ipdb.set_trace()
    #x_emb_r=fluid.layers.sequence_reverse(x_emb, name=None)
    #x_emb_array=numpy.array(x_emb_r)
    #print(x_emb_array)
    #ipdb.set_trace()
    #layers.Print(x_emb_array, message='x_emb_array', summarize=10)
    #layers.Print(x_emb_r, message='x_emb_r', summarize=10)
    fw_hiddens, fw_hiddens_ori = encoder_1(
        x_emb,
        vocab_size,
        emb_size,
        para_name='fw_',
        args=None)
    bw_hiddens, bw_hiddens_ori = encoder_1(
         x_emb_r,
         vocab_size,
         emb_size,
         para_name='bw_',
         args=None)
    embedding=layers.concat(input=[fw_hiddens,bw_hiddens],axis=1)
    if modify==1:
         embedding = dropout(embedding)
    #embedding.stop_gradient=True
    #layers.Print(embedding, message='embedding', summarize=10)
    #a = fluid.layers.create_global_var(shape=[1,], dtype="float32", persistable=False, name='gamma', value=0.0)
    a = layers.create_parameter(
        [1], dtype="float32", name="gamma")
    #layers.Print(a, message='a', summarize=10)
    #embedding = fluid.layers.mul(a,embedding)
    #layers.Print(embedding, message='embedding', summarize=10)
    embedding=embedding*a
    #layers.Print(embedding, message='embedding_1', summarize=10)
    return embedding

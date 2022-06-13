const Tokenizer = require('./Tokenizer');
const Vocab = require('./Vocab');
const Charcode = require('./Charcode');

const fs = require('fs');
const tf = require('@tensorflow/tfjs-node-gpu');

function Sequencializer(options){
    let self = this;
    initialize(self,options);
}

function initialize(self,options){
    options = options || {};
    let seqLength = options.seqLength || 100;
    let text = "";
    let encoder;
    let dataset = null;
    let sequences = null;
    let batchSize = options.batchSize || 64;
    
    switch(options.type){
        case 'tokenizer':
            encoder = new Tokenizer();
            break;
        case 'charcode':
            encoder = new Charcode();
            break;
        default:
            encoder = new Vocab();
            break;
    }

    let loadText = function(t){
        text += t.toString();
        encoder.clear().loadText(text);
        dataset = null;
        sequences = null;
    };

    let clear = function(){
        text = "";
        encoder.clear();
        dataset = null;
        sequences = null;
    };

    let loadTextFile = function(filename){
        loadText(fs.readFileSync(filename,{encoding: 'utf-8'}));
    };

    Object.defineProperty(self,'loadText',{
        get:function(){
            return loadText;
        }
    });

    Object.defineProperty(self,'loadTextFile',{
        get:function(){
            return loadTextFile;
        }
    });

    Object.defineProperty(self,'clear',{
        get:function(){
            return clear;
        }
    });

    Object.defineProperty(self,'seqLength',{
        get:function(){
            return seqLength;
        }
    });

    Object.defineProperty(self,'encoderSize',{
        get:function(){
            return encoder.size;
        }
    });

    Object.defineProperty(self,'dataset',{
        get:function(){ 
            if(dataset === null){
                dataset = tf.data.array(encoder.encode(text));
            }
            return dataset;
        }
    });

    Object.defineProperty(self,'randomSequences',{
        get:function(){
            return self.sequences.shuffle(1024);
        }
    });

    Object.defineProperty(self,'sequences',{
        get:function(){
            if(sequences === null){
                sequences = self.dataset.batch(seqLength+1).map(function(t){
                    let arr = t.arraySync();
                    return [
                        arr.slice(0,arr.length-1),
                        arr.slice(1,arr.length)
                    ];
                }).batch(batchSize).map(function(t){
                    let array = t.arraySync();
                    let xBuffer = tf.buffer([batchSize,seqLength,self.encoderSize]);
                    let yBuffer = tf.buffer([batchSize,self.encoderSize]);
                    for(let i = 0; i < batchSize; i++) {
                        for(let j = 0; j < seqLength; j++) {
                            xBuffer.set(1,i,j,array[i][0][j]);
                        }
                        yBuffer.set(1,i,array[i][1][seqLength-1]);
                    }
                    return {
                        xs:xBuffer.toTensor(),
                        ys:yBuffer.toTensor()
                    };
                });
            }
            return sequences;
        }
    });

    Object.defineProperty(self,'batchSize',{
        get:function(){
            return batchSize;
        }
    });
}



module.exports = Sequencializer;
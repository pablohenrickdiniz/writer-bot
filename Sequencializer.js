const Tokenizer = require('./Tokenizer');
const Vocab = require('./Vocab');
const Charcode = require('./Charcode');

const fs = require('fs');
const tf = require('@tensorflow/tfjs-node-gpu');
const { encode } = require('punycode');

function Sequencializer(options){
    let self = this;
    initialize(self,options);
}

function createSequencesDataset(dataset,seqLength,batchSize,encoderSize){
    return dataset.batch(seqLength+1)
        .filter(function(t){
            return t.shape[0] === seqLength+1;
        })
        .map(function(t){
            let arr = t.arraySync();
            return [
                arr.slice(0,arr.length-1),
                arr.slice(1,arr.length)
            ];
        }).batch(batchSize).map(function(t,b){
            let array = t.arraySync();
            if(array.length < batchSize){
                return null;
            }
            let xBuffer = tf.buffer([batchSize,seqLength,encoderSize]);
            let yBuffer = tf.buffer([batchSize,encoderSize]);
        
            for(let i = 0; i < batchSize; i++) {
                for(let j = 0; j < seqLength; j++) {
                    xBuffer.set(1,i,j,array[i][0][j]);
                }
                yBuffer.set(1,i,array[i][1][seqLength-1]);
            }
            return {
                xs:xBuffer.toTensor(),
                ys:yBuffer.toTensor()
            }
        })
        .filter(function(t){
            return t !== null;
        });
}

function initialize(self,options){
    options = options || {};
    let seqLength = options.seqLength || 128;
    let text = options.text || "";
    let encoder;
    let dataset = null;
    let batchSize = options.batchSize || 64;
    let type = options.type || 'vocab';
    
    switch(type){
        case 'tokenizer':
            encoder = new Tokenizer(options.encoder);
            break;
        case 'charcode':
            encoder = new Charcode(options.encoder);
            break;
        default:
            encoder = new Vocab(options.encoder);
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

    let decodeIndex = function(index){
        return encoder.items[index]?encoder.items[index]:"";
    };

    let decodeItem = function(item){
        let items = encoder.items;
        for(let i = 0; i < items.length;i++){
            if(item === items[i]){
                return i;
            }
        }
        return 0;
    };

    let toJSON = function(){
        return {
            seqLength:seqLength,
            text:text,
            batchSize:batchSize,
            encoder:encoder,
            type:type
        };
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

    Object.defineProperty(self,'trainingDataset',{
        get:function(){
            let size = self.dataset.size;
            let half = Math.floor(size/2);
            return self.dataset.take(half);
        }
    });


    Object.defineProperty(self,'testingDataset',{
        get:function(){
            let size = self.dataset.size;
            let half = Math.floor(size/2);
            return self.dataset.skip(half);
        }
    });

    Object.defineProperty(self,'randomTrainingSequences',{
        get:function(){
            return self.trainingSequences.shuffle(1024);
        }
    });

    Object.defineProperty(self,'randomTestingSequences',{
        get:function(){
            return self.testingSequences.shuffle(1024);
        }
    });

    Object.defineProperty(self,'trainingSequences',{
        get:function(){
            return createSequencesDataset(self.trainingDataset,seqLength,batchSize,self.encoderSize);
        }
    });

    Object.defineProperty(self,'testingSequences',{
        get:function(){
            return createSequencesDataset(self.testingDataset,seqLength,batchSize,self.encoderSize);
        }
    });

    Object.defineProperty(self,'batchSize',{
        get:function(){
            return batchSize;
        }
    });

    Object.defineProperty(self,'decodeIndex',{
        get:function(){
            return decodeIndex;
        }
    });

    Object.defineProperty(self,'decodeItem',{
        get:function(){
            return decodeItem;
        }
    });

    Object.defineProperty(self,'toJSON',{
        get:function(){
            return toJSON;
        }
    });

    Object.defineProperty(self,'encoder',{
        get:function(){
            return encoder;
        }
    });
}



module.exports = Sequencializer;
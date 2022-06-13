
const tf = require('@tensorflow/tfjs-node-gpu');
const Sequencializer = require('./Sequencializer');

function Model(options){
    let self = this;
    initialize(self,options);
}

function initialize(self,options){
   options = options || {}; 
   let seqLength = options.seqLength || 100;
   let hiddenLayers = options.hiddenLayers || 1;
   let units = options.units || 128;
   let model = null;
   let sequencializer = null;

   let train = async function(epochs){
        await self.model.fitDataset(self.sequencializer.sequences.take(100),{
            epochs:epochs || 100
        });
   };

   let generate = function(length){

   };

   let loadTextFile = function(filename){
       self.sequencializer.loadTextFile(filename);
   };

   let loadText= function(text){
        self.sequencializer.loadText(text);
    };

   Object.defineProperty(self,'seqLength',{
        get:function(){
            return seqLength;
        }
   });

    Object.defineProperty(self,'model',{
        get:function(){
            if(model === null){
                model = tf.sequential();
                let seq = self.sequencializer;
                for (let i = 0; i < hiddenLayers; ++i) {
                    model.add(tf.layers.lstm({
                        units: units,
                        returnSequences: i < hiddenLayers-1,
                        inputShape: i === 0 ? [seqLength, seq.encoderSize] : undefined
                    }));
                }
                model.add( tf.layers.dense({units: seq.encoderSize, activation: 'softmax'}));
                model.compile({
                    optimizer: tf.train.rmsprop(0.001), 
                    loss: 'categoricalCrossentropy'
                });
            }
            return model;
        }
    });

   Object.defineProperty(self,'train',{
        get:function(){
            return train;
        }
   });

   Object.defineProperty(self,'generate',{
        get:function(){
            return generate;
        }
    });

    Object.defineProperty(self,'sequencializer',{
        get:function(){
            if(sequencializer === null){
                sequencializer = new Sequencializer({
                    type:options.encoder,
                    seqLength:seqLength
                });
            }
            return sequencializer;
        }
    });

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
}

module.exports = Model;
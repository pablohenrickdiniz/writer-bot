
const tf = require('@tensorflow/tfjs-node-gpu');
const Sequencializer = require('./Sequencializer');

function Model(options){
    let self = this;
    initialize(self,options);
}

function sample(probs, temperature) {
    return tf.tidy(() => {
      const logits = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
      const isNormalized = false;
      return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
    });
}

function initialize(self,options){
   options = options || {}; 
   let seqLength = options.seqLength || 100;
   let hiddenLayers = options.hiddenLayers || 1;
   let units = options.units || 128;
   let model = null;
   let sequencializer = null;

   let train = async function(epochs,callback){
        await self.model.fitDataset(self.sequencializer.randomSequences,{
            epochs:epochs,
            verbose:0,
            callbacks:{
                onEpochEnd:function(epochs,log){
                    callback(epochs+1,log.loss);
                }
            }
        });
   };

   let generate = async function(length){
       let text = "";
       let model = self.model;
       let indexes = await self.sequencializer.dataset.batch(seqLength).map((t) => t.arraySync()).take(1).toArray()[0];
       do{
            let xBuffer = tf.buffer([1,seqLength,self.sequencializer.encoderSize]);
            for(let i = 0;i < seqLength;i++){
                xBuffer.set(1,0,i,indexes[i]);
            }
            let input = xBuffer.toTensor();
            /** tensor de probabilidades */
            let output = model.predict(input).squeeze();
            let index = sample(output,0.6);
            let chr = sequencializer.decodeIndex(index);
            text += chr;
            indexes.shift();
            indexes.push(index);
       }while(text.length < length);
      
       return text;
   };

   let loadTextFile = function(filename){
       self.sequencializer.loadTextFile(filename);
       return self;
   };

   let loadText= function(text){
        self.sequencializer.loadText(text);
        return self;
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
                    optimizer: tf.train.rmsprop(0.01), 
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
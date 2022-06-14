
const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const path = require('path');
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
   let seqLength = options.seqLength || 128;
   let hiddenLayers = options.hiddenLayers || 1;
   let units = options.units || 128;
   let model = null;
   let sequencializer = null;
   let learningRate = 0.001;

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

   let generate = async function(length,callback){
       let model = self.model;
       let indexes = (await self.sequencializer.dataset.batch(seqLength).map((t) => t.arraySync()).take(1).toArray())[0];
       let text = indexes.map((i) => self.sequencializer.decodeIndex(i)).join("");
       if(callback){
          callback(text);
       }
       let count = 0;
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
            if(callback){
                callback(chr);
            }
            indexes.shift();
            indexes.push(index);
            count++;
       }while(count < length);
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

    let save = async function(outputdir){
        let model = self.model;
        if(!fs.existsSync(outputdir)){
            fs.mkdirSync(outputdir,{
                recursive:true
            });
        }
        await model.save('file://'+outputdir);
        fs.writeFileSync(path.join(outputdir,'data.json'),JSON.stringify(this,null,4));
    };

    let load = async function(outputdir){
        if(
            !fs.existsSync(outputdir)
        ){
            return false;
        }
       
        let dataFile = path.join(outputdir,'data.json')

        if(!fs.existsSync(dataFile)){
           return false;
        }

        let modelFile = path.join(outputdir,'model.json');

        if(!fs.existsSync(modelFile)){
           return false;
        }
        model = await tf.loadLayersModel('file://'+modelFile);
        model.compile({
            optimizer: tf.train.rmsprop(learningRate), 
            loss: 'categoricalCrossentropy'
        });
        let tmp = JSON.parse(fs.readFileSync(dataFile,{encoding:'utf-8'}));
        Object.keys(tmp).forEach(function(k){
            switch(k){
                case 'learningRate':
                    learningRate = tmp[k];
                    break;
                case 'hiddenLayers':
                    hiddenLayers = tmp[k];
                    break;
                case 'units':
                    units = tmp[k];
                    break;
                case 'sequencializer':
                    sequencializer = new Sequencializer(tmp[k]);
                    break;
            }
        });
    };

    let toJSON = function(){
        return {
            learningRate:self.learningRate,
            sequencializer:self.sequencializer,
            hiddenLayers:hiddenLayers,
            units:units
        };
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
                    optimizer: tf.train.rmsprop(learningRate), 
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

    Object.defineProperty(self,'save',{
        get:function(){
            return save;
        }
    });

    Object.defineProperty(self,'load',{
        get:function(){
            return load;
        }
    });

    Object.defineProperty(self,'learningRate',{
        get:function(){
            return learningRate;
        }
    });

    Object.defineProperty(self,'toJSON',{
        get:function(){
            return toJSON;
        }
    });
}

module.exports = Model;
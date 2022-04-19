const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');
const fs = require('fs');
const TextUtils = require('./TextUtils');

function TokenNetwork(options){
    let self = this;
    initialize(self);
    options = options || {};
    Object.keys(options).forEach(function(k){
        self[k] = options[k];
    });
}

TokenNetwork.prototype.loadTextFile = function(filename){
    let self = this;
    let contents = fs.readFileSync(filename,{encoding:'utf-8'});
    self.loadText(contents);
    return self;
};

TokenNetwork.prototype.loadText = function(text){
    let self = this;
    text = TextUtils.cleanText(text,self.characters);
    let paragraphs = TextUtils.split(text,self.paragraphSeparators).map(function(paragraph){
        let tokens = [];
        TextUtils.split(paragraph,self.letterSeparators).forEach(function(w){
            tokens = tokens.concat(TextUtils.split(w,self.tokenSeparators,true));
        });
        return tokens;
    });
    self.data.push(paragraphs);
    self.tokens = null;
    return this;
};

TokenNetwork.prototype.getCharactersCount = function(){
    return this.characters.length;
};

TokenNetwork.prototype.getModel = async function(){
    let self = this;
    if(self.model === null){
        let model;
        if(fs.existsSync(self.getModelJsonFile())){
            model = await tf.loadLayersModel('file://'+self.getModelJsonFile());
        }
        else{
            let units = self.getUnits();
            let halfUnits = units/2;
            model = tf.sequential({
                layers:[
                    tf.layers.dense({units:halfUnits,inputShape:[1]}),
                    tf.layers.dense({units:halfUnits}),
                    tf.layers.dense({units:units})
                ]
            });
        }
        model.compile({
            loss:'meanSquaredError',
            optimizer:tf.train.sgd(self.getLearningRate())        
        });
        self.model = model;
    }
    return self.model;
};


TokenNetwork.prototype.train = async function(epochs){
    let self = this;
    let trainingData = self.getTrainigData();
    let x = [];
    let y = [];
    for(let i = 0; i < trainingData.length;i++){
        let t = trainingData[i];
        x.push(t[0]);
        y.push(t[1]);
    }
    epochs = epochs || 100;
    let loss = 0;
    do{
        self.model = null;
        let model = await self.getModel();
        let tx = tf.tensor(x);
        let ty = tf.tensor(y);
        await model.fit(tx,ty,{
            epochs:epochs,
            verbose:0,
            callbacks:{
                onEpochEnd:function(logs,b){
                    loss = b.loss;
                    if(logs % 1 === 0){
                        let log = [[
                            logs.toString().padStart(epochs.toString().length,'0'),
                            epochs
                        ].join('/'),'treinando '+self.name+', loss:'+b.loss].join(' - ');
                        if(!isNaN(loss)){
                            console.log(log);
                        }
                    }
                    if(isNaN(loss)){
                        model.stopTraining = true;
                        self.incrementLearningRate();
                    }
                }
            }
        });
    }while(isNaN(loss));
    if(!isNaN(loss)){
        await self.save();
    }
};

TokenNetwork.prototype.predict = async function(lines){
    let self = this;
    lines = lines.constructor !== [].constructor?[lines]:lines;
    let model = await self.getModel();
    lines = lines.map((l) => this.encodeLine(l));
    let result = model.predict(tf.tensor(lines,[lines.length,1])).arraySync();
    return result.map(function(encoded){
        return self.decodeText(encoded);
    }).join("\n");
};

TokenNetwork.prototype.save = async function(){
    let self = this;
    let model = await self.getModel();
    let modelDir = self.getModelDir();
    if(!fs.existsSync(modelDir)){
        fs.mkdirSync(modelDir,{
            recursive:true
        });
    }
    await model.save('file://'+modelDir);
    self.saveLearningRate();
};

TokenNetwork.prototype.getTokenCode = function(token){
    let self = this;
    let charactersCount = self.charactersCount;
    let prod = token.split('').map((chr) => self.characters.indexOf(chr)).reduce(function(prod,current,index){
        return prod+(current*Math.pow(charactersCount,token.length-index-1));
    },0);
    return prod/self.tokenInterval;
};


function initialize(self){
    let tokenMaxLength = null;
    let tokens = null;
    let characters = [];
    let learningRate = 0.001;
    let letterSeparators = [' '];
    let paragraphSeparators = ['\n'];
    let tokenSeparators = ['.',',',';','!','\'','(',')',':','?'];
    let data = [];
    let tokenInterval = null;


    Object.defineProperty(self,'tokenMaxLength',{
        get:function(){
            if(tokenMaxLength === null){
                tokenMaxLength = self.tokens.reduce((s,t) => Math.max(s,t.length),0);
            }
            return tokenMaxLength;
        }
    });

    Object.defineProperty(self,'tokens',{
        get:function(){
            if(tokens === null){
                tokens = [];
                data.forEach(function(text){
                    text.forEach(function(paragraph){
                        paragraph.forEach(function(token){
                            if(tokens.indexOf(token) === -1){
                                tokens.push(token);
                            }
                        });
                    });
                });
                tokens = tokens.sort(function(a,b){
                    let diff =  a.length - b.length;
                    if(diff === 0){
                        diff = a.localeCompare(b);
                    }
                    return diff;
                });
            }
            return tokens;
        },
        set:function(tks){
            if(!tks){
                tokens = null;
            }
        }
    });

    Object.defineProperty(self,'characters',{
        get:function(){
            return characters;
        },
        set:function(chrs){
            characters = [...new Set(chrs)].sort();
        }
    });

    Object.defineProperty(self,'charactersCount',{
        get:function(){
            return characters.length;
        }
    });
    

    Object.defineProperty(self,'learningRate',{
        get:function(){
            return learningRate;
        },
        set:function(lr){
            learningRate = Math.abs(lr);
        }
    });

    Object.defineProperty(self,'letterSeparators',{
        get:function(){
            return letterSeparators;
        },
        set:function(ls){
            letterSeparators = [...new Set(ls)].sort();
        }
    });

    Object.defineProperty(self,'paragraphSeparators',{
        get:function(){
            return paragraphSeparators;
        },
        set:function(ps){
            paragraphSeparators = [...new Set(ps)].sort();
        }
    });

    Object.defineProperty(self,'tokenSeparators',{
        get:function(){
            return tokenSeparators;
        },
        set:function(ts){
            tokenSeparators = [...new Set(ts)].sort();
        }
    });

    Object.defineProperty(self,'data',{
        get:function(){
            return data;
        }
    });

    Object.defineProperty(self,'tokenInterval',{
        get:function(){
            return Math.pow(self.charactersCount,self.tokenMaxLength);
        }
    });
}

module.exports = TokenNetwork;
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
    self.loadText(fs.readFileSync(filename,{encoding:'utf-8'}));
    return self;
};

TokenNetwork.prototype.loadText = function(text){
    let self = this;
    text = TextUtils.cleanText(text);
    self.characters = self.characters.concat(TextUtils.charactersFromText(text));
    self.data = TextUtils.split(text,self.paragraphSeparators,true).map(function(paragraph){
        return TextUtils.split(paragraph,self.tokenSeparators.concat(self.letterSeparators),true);
    });
    self.tokens = null;
    return this;
};

TokenNetwork.prototype.lineTo3d = function(l){
    let self = this;
    let line = [...l];
    let units = self.units;
    while(line.length < units){
        line.push(' ');
    }
    line = self.encodeLine(line);
    return tf.tensor(line,[1,self.units,1]).arraySync()[0];
};

TokenNetwork.prototype.encodeLine = function(l){
    let self = this;
    return l.map((t) => self.getTokenWeight(t));
};


TokenNetwork.prototype.train = async function(epochs,step){
    let self = this;
    let trainingData = self.trainingData;
    let x = [];
    let y = [];

    for(let i = 0; i < trainingData.length; i++){
        x.push(trainingData[i][0]);
        y.push(trainingData[i][1]);
    }
   
    epochs = epochs || 100;
    let loss = 0;
    do{
        let model = self.model;
        let tx = tf.tensor(x);
        let ty = tf.tensor(y);

        await model.fit(tx,ty,{
            epochs:epochs,
          //  batchSize: 32,
            verbose:0,
            callbacks:{
                onEpochEnd:function(logs,b){
                    loss = b.loss;
                    if(!isNaN(loss) && step){
                        self.loss = loss;
                        step(logs+1,epochs,loss);
                    }
                    else{
                        model.stopTraining = true;
                        self.incrementLearningRate();
                        self.model = null;
                    }
                }
            },
            
        });
    }
    while(isNaN(loss));
};

TokenNetwork.prototype.predict = function(text){
  
};

TokenNetwork.prototype.toJSON = function(){
    let self = this;
    return {
        learningRate:self.learningRate,
        loss:self.loss,
        characters:self.characters,
        letterSeparators:self.letterSeparators,
        paragraphSeparators:self.paragraphSeparators,
        tokenSeparators:self.tokenSeparators
    };
};

TokenNetwork.prototype.save = async function(outputdir){
    let self = this;
    let model = self.model;
    if(!fs.existsSync(outputdir)){
        fs.mkdirSync(outputdir,{
            recursive:true
        });
    }
    await model.save('file://'+outputdir);
    fs.writeFileSync(path.join(outputdir,'data.json'),JSON.stringify(this,null,4));
};

TokenNetwork.prototype.load = async function(outputdir){
    let self = this;
    if(fs.existsSync(outputdir)){
        let dataFile = path.join(outputdir,'data.json')
        if(fs.existsSync(dataFile)){
            let options = JSON.parse(fs.readFileSync(dataFile,{encoding:'utf-8'}));
            Object.keys(options).forEach(function(k){
                self[k] = options[k];
            });
        }
        let modelFile = path.join(outputdir,'model.json');
        if(fs.existsSync(modelFile)){
            let model = await tf.loadLayersModel('file://'+modelFile);
            model.compile({
                loss:tf.losses.meanSquaredError,
                optimizer:tf.train.sgd(self.learningRate)          
            });
            self.model = model;
        }
    }
};

TokenNetwork.prototype.weightToToken = function(weight){
    let self = this;
    let charactersCount = self.charactersCount;
    let prod = self.tokenInterval*weight;
    let indexes = [];
    while(prod > 0){
        indexes.push(prod % charactersCount);
        prod = Math.floor(prod / charactersCount);
    }
    return indexes
        .map(index => self.characters[index]?self.characters[index]:null)
        .filter(chr => chr !== null)
        .join('');
};

TokenNetwork.prototype.getNearestToken = function(weight){
    let self = this;
    let index = parseInt(weight)-1;
    if(index >= 0 && self.tokens[index]){
        return self.tokens[index]
    }
    return null;
    /*
    let self = this; 
    let diff = null;
    let nearest = null;
    let tokensWeights = self.tokensWeights;
    Object.keys(tokensWeights).forEach(function(token){
        let w = tokensWeights[token];
        let d  = Math.abs(weight-w);
        if(diff === null || diff > d){
            diff = d;
            nearest = token;
        }
    });
    return nearest;*/
};

TokenNetwork.prototype.getTokenWeight = function(token){
    let self = this;
    return this.tokens.indexOf(token)+1;
    /*
    let self = this;
    let charactersCount = self.charactersCount;
    let prod = token.split('').map((chr) => self.characters.indexOf(chr)).reduce(function(prod,current,index){
        return prod+(current*Math.pow(charactersCount,token.length-index-1));
    },0);
    return prod/self.tokenInterval;*/
};


TokenNetwork.prototype.incrementLearningRate = function(){
    let self = this;
    let zeros = 0;
    let tmp = String(self.learningRate).split('.');
    if(tmp[1]){
        zeros = tmp[1].length;
    }
    self.learningRate = 1/Math.pow(10,zeros+1);
    return self;
};



function initialize(self){
    let tokenMaxLength = null;
    let tokens = null;
    let tokensWeights = null;
    let characters = [];
    let learningRate = 0.1;
    let loss = null;
    let letterSeparators = [' '];
    let paragraphSeparators = ['\n'];
    let tokenSeparators = ['.',',',';','!','\'','(',')',':','?'];
    let data = [];
    let tokenInterval = null;
    let lineInterval = null;
    let units = null;
    let model = null;

    let reset = function(){
        tokens = null;
        tokensWeights = null;
        tokenMaxLength = null;
        units = null;
        tokenInterval = null;
        lineInterval = null;
    };


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
                tokens = [...new Set(data.reduce((a,b) => a.concat(b)))].sort(function(a,b){
                    let diff =  a.length - b.length;
                    if(diff === 0){
                        diff = a.toString().localeCompare(b.toString());
                    }
                    return diff;
                });
            }
            return tokens;
        },
        set:function(tks){
            if(!tks){
                reset();
            }
        }
    });

    Object.defineProperty(self,'tokensWeights',{
        get:function(){
            if(tokensWeights === null){
                let self = this;
                tokensWeights = [];
                self.tokens.forEach(function(token){
                    tokensWeights[token] = self.getTokenWeight(token);
                });
            }
            return tokensWeights;
        }
    });

    Object.defineProperty(self,'characters',{
        get:function(){
            return characters;
        },
        set:function(chrs){
            characters = [...new Set(chrs)].sort((a,b) => a.charCodeAt(0) - b.charCodeAt(0));
            reset();
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
            learningRate = lr;
        }
    });

    Object.defineProperty(self,'letterSeparators',{
        get:function(){
            return letterSeparators;
        },
        set:function(ls){
            letterSeparators = [...new Set(ls)].sort();
            reset();
        }
    });

    Object.defineProperty(self,'paragraphSeparators',{
        get:function(){
            return paragraphSeparators;
        },
        set:function(ps){
            paragraphSeparators = [...new Set(ps)].sort();
            reset();
        }
    });

    Object.defineProperty(self,'tokenSeparators',{
        get:function(){
            return tokenSeparators;
        },
        set:function(ts){
            tokenSeparators = [...new Set(ts)].sort();
            reset();
        }
    });

    Object.defineProperty(self,'data',{
        get:function(){
            return data;
        },
        set:function(d){
            data = d;
        }
    });


    Object.defineProperty(self,'trainingData',{
        get:function(){
            let trainingData = [];
            for(let i = 0; i < data.length-1;i++){
                let input = self.lineTo3d(data[i],0);
                let output = self.lineTo3d(data[i+1]);
                trainingData.push([
                    input,
                    output
                ]);
            }

            return trainingData;
        }
    });

    Object.defineProperty(self,'tokenInterval',{
        get:function(){
            if(tokenInterval === null){
                tokenInterval = Math.pow(self.charactersCount,self.tokenMaxLength);
            }
            return tokenInterval;
        }
    });

    Object.defineProperty(self,'lineInterval',{
        get:function(){
            if(lineInterval === null){
                let length = self.paragraphs.length;
                let count = 2;
                do{
                    lineInterval = Math.pow(2,count);
                    count++;
                }
                while(lineInterval < length);
            }
            return lineInterval;
        }
    });

    Object.defineProperty(self,'maxParagraphLength',{
        get:function(){
            let max = 0;
            let sum = 0;
            for(let i = 0; i < data.length;i++){
                if(data[i] === "\n"){
                    sum = 0;
                }
                else{
                    sum++;
                }
                max = Math.max(max,sum);
            }
            return max;
        }
    });

    Object.defineProperty(self,'units',{
        get:function(){
            if(units === null){
                let count = 2;
                let length = self.maxParagraphLength;
                let tmp = 2;
                while(tmp < length){
                    tmp = Math.pow(2,count);
                    count++;
                }
                units = tmp;
            }
            return units;
        }
    });

    Object.defineProperty(self,'model',{
        get:function(){
            if(model === null){
                let m = tf.sequential();
                let units = self.units;
                let rnn = tf.layers.simpleRNN({
                    units:1,
                    returnSequences:true,
                    activation:'linear',
                    inputShape:[
                        units,
                        1
                    ]
                });
                m.add(rnn);
                m.add(tf.layers.dense({
                    units:1,
                    inputShape:[units,1],
                    activation:'linear'
                }));
                m.compile({
                    loss:tf.losses.meanSquaredError,
                    optimizer:tf.train.sgd(self.learningRate)        
                });
                model = m;
            }
            return model;
        },
        set:function(m){
            model = m;
        }
    });

    Object.defineProperty(self,'loss',{
        set:function(l){
            if(!isNaN(l)){
                loss = l;
            }
        },
        get:function(){
            return loss;
        }
    });
}

module.exports = TokenNetwork;
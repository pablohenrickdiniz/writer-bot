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


TokenNetwork.prototype.train = async function(epochs,step){
    let self = this;
    let trainingData = self.trainingData;
    let x = [];
    let y = [];
    let lineInterval = self.lineInterval;
    let units = self.units;
    for(let i = 0; i < trainingData.length;i++){
        for(let j = 0; j < trainingData[i].length;j++){
            x.push([i/lineInterval,j/units]);
            y.push(trainingData[i][j]);
        }
        console.log(i*100 / trainingData.length);
    }
    epochs = epochs || 100;
    let loss = 0;
    do{
        let model = self.model;
        let tx = tf.tensor(x,[x.length,2]);
        let ty = tf.tensor(y);
        await model.fit(tx,ty,{
            epochs:epochs,
            verbose:0,
            callbacks:{
                onEpochEnd:function(logs,b){
                    loss = b.loss;
                    if(!isNaN(loss) && step){
                        self.loss = loss;
                        step(logs,epochs,loss);
                    }
                    else{
                        model.stopTraining = true;
                        self.incrementLearningRate();
                        self.model = null;
                    }
                }
            }
        });
    }
    while(isNaN(loss));
};

TokenNetwork.prototype.predict = function(index){
    let self = this;
    let model = self.model;
    let results = [];
    for(let i = 0; i < self.units;i++){
        results.push(model.predict(tf.tensor([[index/self.lineInterval,i/self.units]])).dataSync()[0]);
    }
    return results.filter((w) => w >= 0).map(function(w){
        return self.getNearestToken(w);
    }).join(' ');
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
                loss:'meanSquaredError',
                optimizer:tf.train.sgd(self.learningRate)        
            });
            self.model = model;
        }
    }
};

TokenNetwork.prototype.getNearestToken = function(weight){
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
    return nearest;
};

TokenNetwork.prototype.getTokenWeight = function(token){
    let self = this;
    let charactersCount = self.charactersCount;
    let prod = token.split('').map((chr) => self.characters.indexOf(chr)).reduce(function(prod,current,index){
        return prod+(current*Math.pow(charactersCount,token.length-index-1));
    },0);
    return prod/self.tokenInterval;
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
    let learningRate = 0.001;
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

    Object.defineProperty(self,'paragraphs',{
        get:function(){
            let tmp = [];
            data.forEach(function(paragraphs){
                tmp = tmp.concat(paragraphs);
            });
            return tmp;
        }
    });

    Object.defineProperty(self,'tokens',{
        get:function(){
            if(tokens === null){
                tokens = [];
                self.paragraphs.forEach(function(paragraph){
                    paragraph.forEach(function(token){
                        if(tokens.indexOf(token) === -1){
                            tokens.push(token);
                        }
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
        }
    });

    Object.defineProperty(self,'trainingData',{
        get:function(){
            let trainingData = [];
            let units = self.units;
            data.forEach(function(phrs){
                trainingData = trainingData.concat(phrs.map(function(tks){
                    tks = tks.map((t) => self.getTokenWeight(t));
                    while(tks.length < units){
                        tks.push(-1);
                    }
                    return tks;
                }));
            });
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
            return self.paragraphs.reduce((a,b) => Math.max(a,b.length),0);
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
                let units = self.units;
                let m = tf.sequential();
                m.add(tf.layers.dense({units:1,inputShape:[2]}));
                m.compile({
                    loss:'meanSquaredError',
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
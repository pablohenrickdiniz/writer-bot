let split = require('./TextUtils').split;

function Tokenizer(text,sep){
    let self = this;
    initialize(self,sep);
    if(text){
        self.loadText(text);
    }
}

function initialize(self,sep){
    sep = [...sep || [' ','\n','.',',',';','!','\'','(',')',':','?']];
    let items = [];

    let encode = function(text){
        return split(text,sep).map((c) => items.indexOf(c));
    };

    let decode = function(indexes){
        return indexes.map((i) => items[i]?items[i]:"").join('');
    };

    let loadText = function(text){
        items = [...new Set(items.concat(split(text,sep)))].sort();
    };
    
    Object.defineProperty(self,'items',{
        get:function(){
            return [...items];
        }
    });

    Object.defineProperty(self,'sep',{
        get:function(){
            return [...sep];
        }
    });

    Object.defineProperty(self,'size',{
        get:function(){
            return this.items.length;
        }
    });

    Object.defineProperty(self,'encode',{
        get:function(){return encode;}
    });

    Object.defineProperty(self,'decode',{
        get:function(){return decode;}
    });

    Object.defineProperty(self,'loadText',{
        get:function(){return loadText;}
    });
}

module.exports = Tokenizer;
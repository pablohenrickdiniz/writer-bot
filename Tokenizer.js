let split = require('./TextUtils').split;

function Tokenizer(options){
    let self = this;
    initialize(self,options);
}

function initialize(self,options){
    options = options || {};
    sep = [...options.sep || [' ','\n','.',',',';','!','\'','(',')',':','?']];
    let items = options.items || [];

    /** Encode text*/
    let encode = function(text){
        return split(text,sep).map((c) => items.indexOf(c));
    };

    /** Decode text */
    let decode = function(indexes){
        return indexes.map((i) => items[i]?items[i]:"").join('');
    };

    /** Load unique tokens from text*/
    let loadText = function(text){
        items = [...new Set(items.concat(split(text,sep)))].sort();
        return self;
    };

    let clear = function(){
        items = [];
        return self;
    };

    let toJSON = function(){
        return {
            items:items,
            sep:sep
        };
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

    Object.defineProperty(self,'clear',{
        get:function(){return clear;}
    });

    Object.defineProperty(self,'toJSON',{
        get:function(){ return toJSON;}
    });
}

module.exports = Tokenizer;
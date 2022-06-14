function Vocab(options){
    let self = this;
    initialize(self,options);
}

function initialize(self,options){
    options = options || {};
    let items = options = options.items || [];

    /** Encode text*/
    let encode = function(text){
        return text.split('').map((c) => items.indexOf(c));
    };

    /** Decode text */
    let decode =  function(indexes){
        return indexes.map((i) => items[i]?items[i]:" ").join('');
    };

    /** Load unique characters from text*/
    let loadText = function(text){
        items = [...new Set(items.concat(text.split('')))].sort();
        return self;
    };

    let clear = function(){
        items = [];
        return self;
    };

    let toJSON = function(){
        return {
            items:items
        };
    };
    
    Object.defineProperty(self,'items',{
        get:function(){
            return [...items];
        }
    });

    Object.defineProperty(self,'size',{
        get:function(){
            return items.length;
        }
    });

    Object.defineProperty(self,'encode',{
        get:function(){ return encode;}
    });

    Object.defineProperty(self,'decode',{
        get:function(){ return decode;}
    });

    Object.defineProperty(self,'loadText',{
        get:function(){ return loadText;}
    });

    Object.defineProperty(self,'clear',{
        get:function(){ return clear;}
    });

    Object.defineProperty(self,'toJSON',{
        get:function(){ return toJSON;}
    });
}

module.exports = Vocab;
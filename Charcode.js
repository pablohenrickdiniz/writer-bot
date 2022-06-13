function Charcode(text){
    let self = this;
    initialize(self);
    if(text){
        self.loadText(text);
    }
}

function initialize(self){
    let items = [];


    /** Encode text*/
    let encode = function(text){
        return text.split('').map((c) => c.charCodeAt(0));
    };

    /** Decode text */
    let decode =  function(indexes){
        return indexes.map((i) => String.fromCharCode(i)).join('');
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
}

module.exports = Charcode;
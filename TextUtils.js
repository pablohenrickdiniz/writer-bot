function cleanCharacter(chr,allowedChr){
    if(allowedChr.indexOf(chr) !== -1){
        return chr;
    }
    return '';
}

function cleanWord(word,allowedChr){
    return word.trim().split('').map(function(chr){
        return cleanCharacter(chr,allowedChr);
    }).filter((chr) => chr.length > 0).join('');
}

function cleanLine(text,allowedChr){
    return text.trim().split(/\s+/).map(function(word){
        return cleanWord(word,allowedChr);
    }).filter(function(word){
        return word.length > 0;
    }).join(' ');
}

function cleanText(text,allowedChr){
    return text.trim().split(/[\r\n]+/).map(function(line){
        return cleanLine(line,allowedChr);
    }).filter((l) => l.length > 0).join("\n");
}

function charcodesFromText(text){
    return text.split('').map((c) => c.charCodeAt(0));
}

function textFromCharcodes(charcodes){
    return charcodes.map(function(c){
        return String.fromCharCode(c);
    }).join('');
}

function encodeText(text,divider){
    return charcodesFromText(text).map((c) => c/divider);
}

function decodeText(encoded,multiplier){
    return textFromCharcodes(encoded.map(function(charcode){
        return Math.round(charcode*multiplier);
    }));
}

function charactersFromText(text){
    let chrs = [];
    for(let i = 0; i < text.length;i++){
        let chr = text.charAt(i);
        if(chrs.indexOf(chr) === -1){
            chrs.push(chr);
        }
    }
    return chrs.sort(function(a,b){
        return a.localeCompare(b);
    });
}

function insertBetween(arr,value){
    let tmp = [];
    for(let i = 0; i < arr.length;i++){
        tmp.push(arr[i]);
        if(i < arr.length - 1){
            tmp.push(value);
        }
    }
    return tmp;
}

function split(text,separator,preserveSeparators = false){
    separator = separator || '';
    if(separator.constructor != [].constructor){
        separator = [separator];
    }

    for(let i = 0; i < separator.length;i++){
        let sep = separator[i];
        if(typeof text === 'string'){
            text = text.split(sep);
            if(preserveSeparators){
               text = insertBetween(text,sep);
            }
            text = text.filter((t) => t.length > 0);
        }
        else{
            let tmp = [];
            for(let j = 0; j < text.length;j++){
                let split = text[j].split(sep);
                if(preserveSeparators){
                    split = insertBetween(split,sep);
                }
                tmp = tmp.concat(split);
            }
            text = tmp.filter((t) => t.length > 0);
        }
    }
  
    return text;
}

let TextUtils = {
    cleanText:cleanText,
    cleanLine:cleanLine,
    cleanWord:cleanWord,
    cleanCharacter:cleanCharacter,
    charcodesFromText:charcodesFromText,
    textFromCharcodes:textFromCharcodes,
    encodeText:encodeText,
    decodeText:decodeText,
    charactersFromText:charactersFromText,
    split:split
};

module.exports = TextUtils;
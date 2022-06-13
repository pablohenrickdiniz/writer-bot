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

function split(text,separator){
    separator = separator || '';
    if(separator.constructor != [].constructor){
        separator = [separator];
    }

    for(let i = 0; i < separator.length;i++){
        let sep = separator[i];
        if(typeof text === 'string'){
            text = text.split(sep);
            text = insertBetween(text,sep);
            text = text.filter((t) => t.length > 0);
        }
        else{
            let tmp = [];
            for(let j = 0; j < text.length;j++){
                let split = text[j].split(sep);
                split = insertBetween(split,sep);
                tmp = tmp.concat(split);
            }
            text = tmp.filter((t) => t.length > 0);
        }
    }
  
    return text;
}

let TextUtils = {
    split:split
};

module.exports = TextUtils;
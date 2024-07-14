import fs from "fs"

export function readFileSync(path){
    return fs.readFileSync(path)
}

export function readDirSync(path){
    return fs.readdirSync(path)
}
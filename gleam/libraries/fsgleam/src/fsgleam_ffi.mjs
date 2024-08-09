import fs from "fs"

export function readFileSync(path){
    return fs.readFileSync(path)
}

export function readDirSync(path){
    return fs.readdirSync(path)
}

export function isDir(path){
    try {
        const stats = fs.statSync(path);
        return stats.isDirectory();
      } catch (err) {
        console.error('Error checking directory:', err);
        return false;
      }
}
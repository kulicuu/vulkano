

use std::str::FromStr;


pub fn process_str_ints(input : &str) -> Vec<u32> {
    let start = String::from(input);
    let mut lines = start.lines();
    let mut condition = true;
    let mut ret_vec : Vec<u32> = Vec::new();
    while condition == true {
        let cursor = lines.next();
        if cursor == None {
            condition = false;
        } else {
            let x300 = u32::from_str(cursor.unwrap());
            if x300.is_ok() == true {
                ret_vec.push(x300.unwrap());
            } else {
                println!("error on index parse with");
            }
        }
    }
    ret_vec
}


pub fn find_three_floats(input : &str) -> Option<Vec<f64>> {
    let x300 = String::from(input);
    let x301 : Vec<&str> = x300.split(" ").collect();
    if x301.len() == 3 {
        let x302 = f64::from_str(x301[0]);
        let x303 = f64::from_str(x301[1]);
        let x304 = f64::from_str(x301[2]);
        if (x302.is_ok() == true) && (x304.is_ok() == true) && (x304.is_ok() == true)  {
            return Some(vec!(x302.unwrap(), x303.unwrap(), x304.unwrap()));
        }
        else {
            return None
        }
    } else {
        return None
    }
}


pub fn process_str_floats(input: &str) -> Vec<Vec<f64>> {
    let start = String::from(input);
    let mut lines = start.lines();
    let mut condition = true;
    let mut ret_vec : Vec<Vec<f64>> = Vec::new();
    while condition == true {
        let cursor = lines.next();
        if cursor == None {
            condition = false;
        } else {
            // println!("The line: {:?}", cursor.unwrap());
            let x200 = find_three_floats(&cursor.unwrap());
            if x200 != None {
                ret_vec.push(x200.unwrap());
            }
        }
    }
    ret_vec
}

fn main() {
    let a = vec!["a", "b", "c"];
    let b = vec!["a", "c"];

    let result = create_comma_separated_string(&a, &b);
    println!("{}", result);
}

fn create_comma_separated_string(a: &[&str], b: &[&str]) -> String {
    let mut d = String::new();

    for e in b {
        if a.contains(e) {
            d.push_str(e);
            d.push(',');
        }
    }

    d
}

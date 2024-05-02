#include <stdio.h>

void print_enum_tag(long *n) {
    printf("%ld\n", *n);
}

void print_i64(long n) {
    printf("%ld\n", n);
}

void print_f64(double n) {
    printf("%f\n", n);
}

void print_str(char *s, long len) {
    printf("%.*s", (int)len, s);
}

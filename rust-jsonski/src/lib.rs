use std::ffi::{CString};
use libc::c_void;

#[derive(Clone, Copy)]
pub struct JsonSkiRecord {
    record: *const c_void
}

mod jsonski_extern {
    use libc::{c_long, c_char, c_void};

    extern "C" {
        pub(crate) fn loadFile(file_name: *const c_char) -> *const c_void;
        pub(crate) fn runJsonSki(query: *const c_char, record: *const c_void) -> c_long;
    }
}

pub fn load_jsonski_record(file_name: &str) -> JsonSkiRecord {
    let c_file_name = CString::new(file_name).unwrap();

    unsafe {
        let record_ptr = jsonski_extern::loadFile(c_file_name.as_ptr());
        JsonSkiRecord {
            record: record_ptr
        }
    }
}

pub fn call_jsonski(query: &str, record: JsonSkiRecord) -> i64 {
    let c_query = CString::new(query).unwrap();

    unsafe {
        jsonski_extern::runJsonSki(c_query.as_ptr(), record.record)
    }
}

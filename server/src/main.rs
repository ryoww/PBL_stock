use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server, StatusCode};
use std::convert::Infallible;

async fn handle_request(req: Request<Body>) -> Result<Response<Body>, Infallible> {
    if req.method() == hyper::Method::POST {
        let body_bytes = hyper::body::to_bytes(req.into_body()).await.unwrap();

        let body_string = String::from_utf8(body_bytes.to_vec()).unwrap();

        println!("Received : {}", body_string);

        Ok(Response::new(Body::from("Received")))
    } else {
        let mut not_allowed = Response::default();
        *not_allowed.status_mut() = StatusCode::METHOD_NOT_ALLOWED;
        Ok(not_allowed)
    }
}

#[tokio::main]
async fn main() {
    let addr = ([127, 0, 0, 1], 3000).into();
    let service = make_service_fn(|_| async { Ok::<_, Infallible>(service_fn(handle_request)) });
    let server = Server::bind(&addr).serve(service);

    println!("listening on http://{}", addr);

    if let Err(e) = server.await {
        eprintln!("server error : {}", e);
    }
}

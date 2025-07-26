import grpc
from telemetry_pb2_grpc import TelemetryServiceStub
from telemetry_pb2 import SubscriptionRequest

def run_telemetry_client():
    channel = grpc.insecure_channel('localhost:50051')
    stub = TelemetryServiceStub(channel)

    # Create a subscription request
    request = SubscriptionRequest(
        client_id="test_client",
        topics=["topic1", "topic2"]
    )

    # Send the subscription request
    response = stub.Subscribe(request)
    print("Subscription response:", response)

if __name__ == "__main__":
    run_telemetry_client()
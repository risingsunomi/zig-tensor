const std = @import("std");
const testing = std.testing;
const tensor = @import("tensor.zig");

// tensor.zig tests
test "test createTensor" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var allocator = gpa.allocator();
    defer {
        _ = gpa.deinit();
    }

    var tensor_shape = [_]usize{ 2, 3, 4 };

    const new_tensor = try tensor.createTensor(&allocator, .int, tensor_shape[0..]);
    defer allocator.free(new_tensor.data);

    std.debug.print("new_tensor.shape: {d}\n", .{new_tensor.shape});

    for (new_tensor.shape, 0..) |dim, i| {
        try testing.expect(dim == tensor_shape[i]);
    }
    try testing.expect(new_tensor.dtype == .int);
}

test "test broadcastShapes" {
    var allocator = testing.allocator;

    var a_shape = [_]usize{ 3, 1 };
    var a = try tensor.createTensor(&allocator, .int, a_shape[0..]);
    defer a.deinit();

    var b_shape = [_]usize{ 2, 3 };
    var b = try tensor.createTensor(&allocator, .int, b_shape[0..]);
    defer b.deinit();

    try tensor.broadcastShapes(&a, &b);

    std.debug.print("a.shape: {d}\n", .{a.shape});

    for (a.shape, 0..) |dim, i| {
        try testing.expect(a.shape[i] == dim);
    }
}

//
//test "test addTensors" {
//    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//    var allocator = gpa.allocator();
//    defer {
//        _ = gpa.deinit();
//    }
//
//    var a_tensor_shape = [_]usize{ 2, 3, 4 };
//    var a = try createTensor(&allocator, .int, a_tensor_shape[0..]);
//    defer allocator.free(a.data);
//
//    var b_tensor_shape = [_]usize{ 3, 4 };
//    var b = try createTensor(&allocator, .int, b_tensor_shape[0..]);
//    defer allocator.free(b.data);
//
//    const out = try addTensors(&a, &b, &allocator);
//    defer allocator.free(out.data);
//
//    try testing.expect(out.shape == a.shape);
//    try testing.expect(out.dtype == .int);
//}

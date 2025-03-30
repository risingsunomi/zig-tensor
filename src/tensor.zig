const std = @import("std");

const Tensor = struct {
    const Dtype = enum { int, float, bool };

    data: []u8,
    shape: []usize,
    dtype: Dtype,
    allocator: *std.mem.Allocator,

    pub fn numel(self: *Tensor) usize {
        var total_elements = 1;
        for (self.shape) |dim| {
            total_elements *= dim;
        }
        return total_elements;
    }

    pub fn dt_size(self: *Tensor) usize {
        return self.numel() * switch (self.dtype) {
            .int, .float => @sizeOf(i64),
            .bool => @sizeOf(bool),
        };
    }

    pub fn reshape(self: *Tensor, new_shape: []usize) !Tensor {
        const new_numel = new_shape.len;
        if (self.numel() != new_numel) {
            return error.OutOfRange;
        }

        return Tensor{
            .data = self.data,
            .shape = new_shape,
            .dtype = self.dtype,
        };
    }

    // deinitialize tensor
    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
    }
};

pub fn createTensor(
    allocator: *std.mem.Allocator,
    dtype: Tensor.Dtype,
    shape: []usize,
) !Tensor {
    // get full shape
    var total_elements: usize = 1;
    for (shape) |dim| {
        total_elements *= dim;
    }

    const element_size: usize = switch (dtype) {
        .int, .float => @sizeOf(i64),
        .bool => @sizeOf(bool),
    };
    const total_size = total_elements * element_size;
    std.debug.print("total_size: {d}\n", .{total_size});

    // allocate mem for data
    var mem = try allocator.alloc(u8, total_size);

    return Tensor{
        .data = mem[0..total_size],
        .shape = shape,
        .dtype = dtype,
        .allocator = allocator,
    };
}

// broadcastShapes
// takes two shapes and broadcasts them to the same shape
pub fn broadcastShapes(
    a: *Tensor,
    b: *Tensor,
) anyerror!void {
    // if same shape no broadcast needed
    if (a.shape.len == b.shape.len) {
        std.debug.print("same shape, no broadcast\n", .{});
        return;
    }

    // place holder while logic is worked out
    return;
}

pub fn addTensors(
    a: *Tensor,
    b: *Tensor,
    allocator: *std.mem.Allocator,
) !Tensor {
    const out_shape = try broadcastShapes(a.shape, b.shape, allocator);

    const out = try createTensor(allocator, a.dtype, out_shape) catch |err| {
        std.debug.print("error: {any}\n", .{err});
        return error.OutOfMemory;
    };

    const total_elements = out.numel();
    const block_size = 64; // Adjust block size based on cache line size and L2 cache size

    var i: usize = 0;
    while (i < total_elements) : (i += block_size) {
        const end = std.math.min(i + block_size, total_elements);

        var j: usize = i;
        while (j < end) : (j += 1) {
            const a_idx = j % a.numel();
            const b_idx = j % b.numel();
            const out_idx = j % out.numel();

            // get the value of a at the index
            const a_val = switch (a.dtype) {
                .int => @as(i64, a.data[a_idx .. a_idx + @sizeOf(i64)]),
                .float => @as(f64, a.data[a_idx .. a_idx + @sizeOf(f64)]),
                .bool => @as(bool, a.data[a_idx .. a_idx + @sizeOf(bool)]),
            };

            // get the value of b at the index
            const b_val = switch (b.dtype) {
                .int => @as(i64, b.data[b_idx .. b_idx + @sizeOf(i64)]),
                .float => @as(f64, b.data[b_idx .. b_idx + @sizeOf(f64)]),
                .bool => @as(bool, b.data[b_idx .. b_idx + @sizeOf(bool)]),
            };

            // perform the operation based on the data type
            const out_val = switch (a.dtype) {
                .int => a_val + b_val,
                .float => a_val + b_val,
                .bool => a_val or b_val,
            };

            std.debug.print(
                "a_val: {d}\nb_val: {d}\nout_val: {d}\n",
                .{ a_val, b_val, out_val },
            );

            // copy the value to the output tensor
            switch (a.dtype) {
                .int => std.mem.copy(
                    u8,
                    out.data[out_idx .. out_idx + @sizeOf(i64)],
                    @as(u8, out_val),
                ),
                .float => std.mem.copy(
                    u8,
                    out.data[out_idx .. out_idx + @sizeOf(f64)],
                    @as(u8, out_val),
                ),
                .bool => std.mem.copy(
                    u8,
                    out.data[out_idx .. out_idx + @sizeOf(bool)],
                    @as(u8, out_val),
                ),
            }
        }
    }

    return out;
}

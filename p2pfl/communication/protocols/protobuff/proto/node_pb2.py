# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: node.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'node.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nnode.proto\x12\x04node\x1a\x1bgoogle/protobuf/empty.proto\"\xd8\x01\n\x0bRootMessage\x12\x0e\n\x06source\x18\x01 \x01(\t\x12\x12\n\x05round\x18\x02 \x01(\x05H\x01\x88\x01\x01\x12\x0b\n\x03\x63md\x18\x03 \x01(\t\x12-\n\x0egossip_message\x18\x04 \x01(\x0b\x32\x13.node.GossipMessageH\x00\x12-\n\x0e\x64irect_message\x18\x05 \x01(\x0b\x32\x13.node.DirectMessageH\x00\x12 \n\x07weights\x18\x06 \x01(\x0b\x32\r.node.WeightsH\x00\x42\x0e\n\x0cpayload_typeB\x08\n\x06_round\"8\n\rGossipMessage\x12\x0b\n\x03ttl\x18\x01 \x01(\x05\x12\x0c\n\x04hash\x18\x02 \x01(\x03\x12\x0c\n\x04\x61rgs\x18\x03 \x03(\t\"\x1d\n\rDirectMessage\x12\x0c\n\x04\x61rgs\x18\x03 \x03(\t\"E\n\x07Weights\x12\x0f\n\x07weights\x18\x01 \x01(\x0c\x12\x14\n\x0c\x63ontributors\x18\x02 \x03(\t\x12\x13\n\x0bnum_samples\x18\x03 \x01(\x05\" \n\x10HandShakeRequest\x12\x0c\n\x04\x61\x64\x64r\x18\x01 \x01(\t\"S\n\x0fResponseMessage\x12\x15\n\x08response\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x12\n\x05\x65rror\x18\x02 \x01(\tH\x01\x88\x01\x01\x42\x0b\n\t_responseB\x08\n\x06_error2\xba\x01\n\x0cNodeServices\x12:\n\thandshake\x12\x16.node.HandShakeRequest\x1a\x15.node.ResponseMessage\x12<\n\ndisconnect\x12\x16.node.HandShakeRequest\x1a\x16.google.protobuf.Empty\x12\x30\n\x04send\x12\x11.node.RootMessage\x1a\x15.node.ResponseMessageb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'node_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_ROOTMESSAGE']._serialized_start=50
  _globals['_ROOTMESSAGE']._serialized_end=266
  _globals['_GOSSIPMESSAGE']._serialized_start=268
  _globals['_GOSSIPMESSAGE']._serialized_end=324
  _globals['_DIRECTMESSAGE']._serialized_start=326
  _globals['_DIRECTMESSAGE']._serialized_end=355
  _globals['_WEIGHTS']._serialized_start=357
  _globals['_WEIGHTS']._serialized_end=426
  _globals['_HANDSHAKEREQUEST']._serialized_start=428
  _globals['_HANDSHAKEREQUEST']._serialized_end=460
  _globals['_RESPONSEMESSAGE']._serialized_start=462
  _globals['_RESPONSEMESSAGE']._serialized_end=545
  _globals['_NODESERVICES']._serialized_start=548
  _globals['_NODESERVICES']._serialized_end=734
# @@protoc_insertion_point(module_scope)

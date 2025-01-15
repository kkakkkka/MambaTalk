import copy
import pickle
import torch
import torch.nn as nn

from .motion_encoder import * 
from .layers import *
from .mamba_block import MambaModel, MambaScan    


class GlobalScan(nn.Module):
    def __init__(self, args):
        super(GlobalScan, self).__init__()
        self.args = args
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 3
        args_top.vae_length = args.motion_f
        args_top.vae_test_dim = args.pose_dims+3+4 
        # learnable query      
        self.learnable_query = nn.Parameter(torch.zeros(1, 1, self.args.pose_dims+3+4))
        self.motion_encoder = VQEncoderV6(args_top)
        self.motion_proj1 = MLP(args.motion_f, args.hidden_size, args.motion_f)
        self.motion_proj2 = nn.Linear(args.motion_f, args.hidden_size)

        # self-attention
        self.position_embeddings = PeriodicPositionalEncoding(args.hidden_size, period=args.pose_length, max_seq_len=args.pose_length)
        self.face_q_self_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=4, dim_feedforward=args.hidden_size, batch_first=True), num_layers=1)
        self.body_q_self_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=4, dim_feedforward=args.hidden_size, batch_first=True), num_layers=1)
        self.motion_self_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=4, dim_feedforward=args.hidden_size, batch_first=True), num_layers=1)
        
        # scan
        self.face_scan = MambaScan(in_dim=args.audio_f, d_intermediate=args.audio_f * 2, out_dim=args.audio_f, n_layer=1)
        self.body_scan = MambaScan(in_dim=args.motion_f, d_intermediate=args.hidden_size, out_dim=args.motion_f, n_layer=1)        
        self.out = nn.Linear(args.audio_f + args.motion_f, args.hidden_size)

    def forward(self, in_motion, mask, speaker_embedding_face, speaker_embedding_body, face_features):
        # learnable query
        learnable_query = self.learnable_query.expand_as(in_motion)
        learnable_query = torch.where(mask == 1, learnable_query, in_motion)
        body_features = self.motion_encoder(learnable_query)
        motion_features = self.motion_proj2(self.motion_proj1(body_features))
        
        # self-attention
        speaker_embedding_face = self.face_q_self_attn(speaker_embedding_face)
        speaker_embedding_body = self.body_q_self_attn(speaker_embedding_body)
        motion_features = speaker_embedding_body + motion_features
        motion_features = self.position_embeddings(motion_features)
        global_motion = self.motion_self_encoder(motion_features)         
        
        # scan
        face_features = self.face_scan(face_features)
        body_features = self.body_scan(body_features)
        global_features = self.out(torch.cat([face_features, body_features], dim=2))
        
        return global_motion, global_features


class LocalScan(nn.Module):
    def __init__(self, args):
        super(LocalScan, self).__init__()
        self.args = args

        # cross-attention
        self.position_embeddings = PeriodicPositionalEncoding(args.hidden_size, period=args.pose_length, max_seq_len=args.pose_length)
        self.motion_cross_attn = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=args.hidden_size, nhead=4, dim_feedforward=args.hidden_size, batch_first=True), num_layers=8)
        self.motion2latent_upper = MLP(args.hidden_size, args.hidden_size, args.hidden_size)
        self.motion2latent_hands = MLP(args.hidden_size, args.hidden_size, args.hidden_size)
        self.motion2latent_lower = MLP(args.hidden_size, args.hidden_size, args.hidden_size)
        self.face_proj = nn.Linear(2*args.hidden_size, args.hidden_size)
        self.upper_proj = nn.Linear(2*args.hidden_size, args.hidden_size)
        self.hands_proj = nn.Linear(2*args.hidden_size, args.hidden_size)
        self.lower_proj = nn.Linear(2*args.hidden_size, args.hidden_size)
        self.body_proj = nn.Linear(args.audio_f, args.hidden_size)

        # scan
        self.face_scan = MambaScan(in_dim=args.hidden_size, d_intermediate=args.hidden_size, out_dim=args.hidden_size, n_layer=4)
        self.upper_scan = MambaModel(d_model=args.hidden_size, n_layer=1, d_intermediate=args.hidden_size)
        self.hands_scan = MambaModel(d_model=args.hidden_size, n_layer=1, d_intermediate=args.hidden_size)
        self.lower_scan = MambaModel(d_model=args.hidden_size, n_layer=1, d_intermediate=args.hidden_size)

        # decoder
        self.motion_down_face = nn.Linear(args.hidden_size, args.vae_codebook_size)
        self.motion_down_upper = nn.Linear(args.hidden_size, args.vae_codebook_size)
        self.motion_down_hands = nn.Linear(args.hidden_size, args.vae_codebook_size)
        self.motion_down_lower = nn.Linear(args.hidden_size, args.vae_codebook_size)
        self.face_classifier = MLP(args.vae_codebook_size, args.hidden_size, args.vae_codebook_size)
        self.upper_classifier = MLP(args.vae_codebook_size, args.hidden_size, args.vae_codebook_size)
        self.hands_classifier = MLP(args.vae_codebook_size, args.hidden_size, args.vae_codebook_size)
        self.lower_classifier = MLP(args.vae_codebook_size, args.hidden_size, args.vae_codebook_size)

    def forward(self, global_motions, global_features, face_latent_in, speaker_embedding_body, body_features, use_word):
        # cross attn
        motion_refined_embeddings_in = global_motions + speaker_embedding_body
        motion_refined_embeddings_in = self.position_embeddings(global_motions)        
        learnable_querys = self.motion_cross_attn(tgt=motion_refined_embeddings_in, memory=self.body_proj(body_features))
        global_motions = global_motions + learnable_querys
        
        # feedforward
        upper_latent = self.motion2latent_upper(global_motions)
        hands_latent = self.motion2latent_hands(global_motions)
        lower_latent = self.motion2latent_lower(global_motions)
        
        upper_latent_in = upper_latent + speaker_embedding_body
        hands_latent_in = hands_latent + speaker_embedding_body
        lower_latent_in = lower_latent + speaker_embedding_body        
        
        upper_latent_in = self.position_embeddings(upper_latent_in)
        hands_latent_in = self.position_embeddings(hands_latent_in)
        lower_latent_in = self.position_embeddings(lower_latent_in)
        face_latent_in = self.position_embeddings(face_latent_in)        
        
        # local scan
        decoded_face = self.face_scan(self.face_proj(torch.cat([face_latent_in, global_features], dim=-1)))          
        motion_upper = self.upper_scan(self.upper_proj(torch.cat([upper_latent_in, hands_latent+lower_latent], dim=-1)))
        motion_hands = self.hands_scan(self.hands_proj(torch.cat([hands_latent_in, upper_latent+lower_latent], dim=-1)))
        motion_lower = self.lower_scan(self.lower_proj(torch.cat([lower_latent_in, upper_latent+hands_latent], dim=-1)))
        
        # codebook
        face_latent = self.motion_down_face(decoded_face)
        upper_latent = self.motion_down_upper(motion_upper+upper_latent)
        hands_latent = self.motion_down_hands(motion_hands+hands_latent)
        lower_latent = self.motion_down_lower(motion_lower+lower_latent)

        # index loss
        cls_face = self.face_classifier(face_latent)
        cls_lower = self.lower_classifier(lower_latent)
        cls_upper = self.upper_classifier(upper_latent)
        cls_hands = self.hands_classifier(hands_latent)

        return {
            "rec_face": face_latent,
            "rec_upper": upper_latent,
            "rec_lower": lower_latent,
            "rec_hands": hands_latent,
            "cls_face": cls_face,
            "cls_upper": cls_upper,
            "cls_lower": cls_lower,
            "cls_hands": cls_hands,
        }


class MambaTalk(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args   
        with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_face = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding), freeze=args.t_fix_pre)
        self.text_encoder_face = nn.Linear(300, args.audio_f) 
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding), freeze=args.t_fix_pre)
        self.text_encoder_body = nn.Linear(300, args.audio_f) 

        # audio/text fusion
        self.audio_pre_encoder_face = WavEncoder(args.audio_f, audio_in=3)
        self.audio_pre_encoder_body = WavEncoder(args.audio_f, audio_in=2)        
        self.at_attn_face = nn.Linear(args.audio_f*2, args.audio_f*2)
        self.at_attn_body = nn.Linear(args.audio_f*2, args.audio_f*2)
        
        # mamba scan
        self.global_scan = GlobalScan(args)
        self.local_scan = LocalScan(args)

        # id emb
        self.speaker_encoder_face = nn.Embedding(25, args.hidden_size)
        self.speaker_encoder_body = nn.Embedding(25, args.hidden_size)

    def forward(self, in_audio=None, in_word=None, mask=None, is_test=None, in_motion=None, use_attentions=True, use_word=True, in_id=None):
        in_word_face = self.text_pre_encoder_face(in_word)
        in_word_face = self.text_encoder_face(in_word_face)
        in_word_body = self.text_pre_encoder_body(in_word)
        in_word_body = self.text_encoder_body(in_word_body)
        bs, t, c = in_word_face.shape

        in_audio_face = self.audio_pre_encoder_face(in_audio)
        in_audio_body = self.audio_pre_encoder_body(in_audio[...,:2])
        
        if in_audio_face.shape[1] != in_motion.shape[1]:
            diff_length = in_motion.shape[1] - in_audio_face.shape[1]
            if diff_length < 0:
                in_audio_face = in_audio_face[:, :diff_length, :]
                in_audio_body = in_audio_body[:, :diff_length, :]
            else:
                in_audio_face = torch.cat((in_audio_face, in_audio_face[:,-diff_length:]),1)
                in_audio_body = torch.cat((in_audio_body, in_audio_body[:,-diff_length:]),1)

        # audio/text fusion
        alpha_at_face = torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c*2)
        alpha_at_face = self.at_attn_face(alpha_at_face).reshape(bs, t, c, 2)
        alpha_at_face = alpha_at_face.softmax(dim=-1)
        fusion_face = in_word_face * alpha_at_face[:,:,:,1] + in_audio_face * alpha_at_face[:,:,:,0]        
        alpha_at_body = torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c*2)
        alpha_at_body = self.at_attn_body(alpha_at_body).reshape(bs, t, c, 2)
        alpha_at_body = alpha_at_body.softmax(dim=-1)
        fusion_body = in_word_body * alpha_at_body[:,:,:,1] + in_audio_body * alpha_at_body[:,:,:,0]
        
        # ID Emd
        speaker_embedding_face = self.speaker_encoder_face(in_id).squeeze(2)
        speaker_embedding_body = self.speaker_encoder_body(in_id).squeeze(2)
        
        # Global Scan
        global_motions, global_features = self.global_scan(in_motion, mask, speaker_embedding_face, speaker_embedding_body, fusion_face)
        
        # Local Scan
        output = self.local_scan(global_motions, global_features, speaker_embedding_face, speaker_embedding_body, fusion_body, use_word)
        
        return output